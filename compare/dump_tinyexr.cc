// The C++ half of the comparison with tinyexr (https://github.com/syoyo/tinyexr): makes .exr
// files of synthetic pixels with it, reads them back with it, and prints a running hash of the
// files or of what it reads. examples/dump.rs prints the same from the Rust port, and
// compare/compare.sh compares the two.
//
//   dump tinyexr_write <seed> <count>         synthetic images of every kind tinyexr writes
//   dump tinyexr_write_small <seed> <count>   (scanlines and tiles, with mip and rip levels,
//                                             three types of value, eight compressions, with
//                                             attributes of their own), saved: the bytes of
//                                             each file, or tinyexr's message where it does
//                                             not save one; and for every fourth, floats of
//                                             one, three or four to a pixel saved the short
//                                             way, as halves or as floats
//   dump tinyexr_parts <seed> <count> <file>   files of one to four parts, each made as for
//                                              tinyexr_write_small and named (now and then
//                                              not, or as the one before), saved: the bytes
//                                              or the message; a third of them damaged, and
//                                              read in a program of their own, with zeros
//                                              and with ones past the end (to the file, with
//                                              whether that made a difference, or stopped
//                                              tinyexr)
//   dump tinyexr_deep <seed> <count> <file>    synthetic deep files of scanlines and of tiles,
//                                              a third of them damaged, read the same way
//   dump tinyexr_zfp_write <seed> <count>        as tinyexr_write and tinyexr_read, of images
//   dump tinyexr_zfp_read <seed> <count> <file>  made for ZFP compression: of floats, most of
//                                                whole blocks of four by four, with the
//                                                attributes of its settings
//   dump tinyexr_read <seed> <count> <file>         synthetic images of every kind tinyexr writes
//   dump tinyexr_read_small <seed> <count> <file>   (scanlines and tiles, with mip and rip
//                                                   levels, three types of value, eight
//                                                   compressions), saved; now and then the
//                                                   window moved or the order of the lines
//                                                   named the other way; a third of them
//                                                   damaged. Its version and its header are
//                                                   read, its image as it is stored or with
//                                                   halves as floats, and it as floats of red,
//                                                   green, blue and alpha; and it is written to
//                                                   the file, with how it was read.
//
// Built without threads and without OpenMP, with ZFP compression against the zfp library
// v1.0.1, against the miniz that src/miniz is the port of,
// and with memory that starts as zeros: tinyexr leaves what a damaged file does not fill as
// malloc gave it.
//
// tinyexr's PIZ writer gives a block 8192 bytes and twice the block's own, and writes past
// that for a small block of many different values (the bitmap alone can take 8192 bytes, and
// the table of code lengths comes after it). Everything allocated with new here is given
// 65536 bytes more than was asked for, so that what is written past the end stays in memory
// the program owns, and the block comes out as it would with room enough.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

// What tinyexr allocates starts as zeros. An allocation of `g_pad_for` bytes (the file tinyexr
// reads a deep image from) starts a page, and is followed by `g_pad` bytes of `g_pad_byte`
// and on to the end of a page (4096 bytes): what a reader that reads past the end of the
// file finds there. Before it, and after that, is a gigabyte of memory that cannot be read,
// so that a reader that reads before the file, or far past its end, stops. (The port refuses
// such a file; it takes what is as far past the end as here to be all zeros and then all
// ones.)
static std::size_t g_pad = 0, g_pad_for = 0;
static unsigned char g_pad_byte = 0;
struct Padded
{
    void* memory;
    unsigned char* base;
    std::size_t total;
};
static std::vector<Padded> g_padded;
// The size of each allocation made while a deep file is read: a damaged file can have a line
// of other than the count of samples its table says, and only the allocation says which.
static std::unordered_map<void*, std::size_t> g_sizes;
static void* HarnessMalloc(std::size_t size)
{
    if (!g_pad || size != g_pad_for)
    {
        void* memory = std::calloc(1, size);
        if (g_pad && memory)
            g_sizes[memory] = size;
        return memory;
    }
    const std::size_t page = 4096;
    if (static_cast<std::size_t>(sysconf(_SC_PAGESIZE)) != page)
    {
        std::fprintf(stderr, "the deep files are compared only where a page is 4096 bytes\n");
        std::abort();
    }
    const std::size_t guard = static_cast<std::size_t>(1) << 30;
    const std::size_t usable = (size + g_pad + page - 1) / page * page;
    const std::size_t total = guard + usable + guard;
    void* base = mmap(NULL, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED)
        return NULL;
    unsigned char* memory = static_cast<unsigned char*>(base) + guard;
    if (mprotect(memory, usable, PROT_READ | PROT_WRITE) != 0)
        std::abort();
    std::memset(memory + size, g_pad_byte, usable - size);
    g_padded.push_back(Padded { memory, static_cast<unsigned char*>(base), total });
    return memory;
}
static void HarnessFree(void* memory)
{
    g_sizes.erase(memory);
    for (std::size_t k = 0; k < g_padded.size(); ++k)
    {
        if (g_padded[k].memory == memory)
        {
            munmap(g_padded[k].base, g_padded[k].total);
            g_padded.erase(g_padded.begin() + static_cast<std::ptrdiff_t>(k));
            return;
        }
    }
    std::free(memory);
}

// (Without __linux__ and __unix__, tinyexr reads a file into memory it allocates, rather than
// mapping it, so that what is outside it is what is put there here.)
#pragma push_macro("__linux__")
#pragma push_macro("__unix__")
#undef __linux__
#undef __unix__
#define malloc(size) HarnessMalloc(size)
#define free(memory) HarnessFree(memory)
#define TINYEXR_USE_MINIZ 1
#define TINYEXR_USE_THREAD 0
#define TINYEXR_USE_OPENMP 0
#define TINYEXR_USE_PIZ 1
#define TINYEXR_USE_ZFP 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#undef malloc
#undef free
#pragma pop_macro("__linux__")
#pragma pop_macro("__unix__")

// (And no more than a gigabyte at once, which the port takes to be more than there may be: a
// file that has tinyexr ask for more stops it.)
void* operator new(std::size_t size)
{
    if (size > (static_cast<std::size_t>(1) << 30))
        throw std::bad_alloc();
    void* memory = std::malloc(size + 65536);
    if (!memory)
        throw std::bad_alloc();
    return memory;
}
void operator delete(void* memory) noexcept
{
    std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}

namespace
{
// The two programs make the same data from the same seed: a 64-bit linear congruential
// generator, the high half of its state.
std::uint64_t g_state;
std::uint32_t Next()
{
    g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(g_state >> 32);
}

// FNV-1a, 64 bits.
std::uint64_t g_hash = 14695981039346656037ull;
void Add(const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i)
    {
        g_hash ^= bytes[i];
        g_hash *= 1099511628211ull;
    }
}
void AddNumber(std::uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = static_cast<unsigned char>(value);
    bytes[1] = static_cast<unsigned char>(value >> 8);
    bytes[2] = static_cast<unsigned char>(value >> 16);
    bytes[3] = static_cast<unsigned char>(value >> 24);
    Add(bytes, 4);
}
void AddFloat(float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    AddNumber(bits);
}
// A string with its end.
void AddString(const char* text)
{
    Add(text, std::strlen(text) + 1);
}

// A name with its end, where it is UTF-8; a mark in its place where it is not (the port keeps
// names as strings, and a name that is not UTF-8 loses the bytes that are not).
void AddName(const char* text)
{
    const auto* s = reinterpret_cast<const unsigned char*>(text);
    bool valid = true;
    for (std::size_t i = 0; s[i] && valid;)
    {
        const unsigned char b = s[i];
        std::size_t more;
        std::uint32_t low = 0x80, high = 0xbf;
        if (b < 0x80)
            more = 0;
        else if (b >= 0xc2 && b <= 0xdf)
            more = 1;
        else if (b >= 0xe0 && b <= 0xef)
        {
            more = 2;
            if (b == 0xe0)
                low = 0xa0;
            if (b == 0xed)
                high = 0x9f;
        }
        else if (b >= 0xf0 && b <= 0xf4)
        {
            more = 3;
            if (b == 0xf0)
                low = 0x90;
            if (b == 0xf4)
                high = 0x8f;
        }
        else
        {
            valid = false;
            break;
        }
        ++i;
        for (std::size_t k = 0; k < more; ++k, ++i)
        {
            if (s[i] < low || s[i] > high)
            {
                valid = false;
                break;
            }
            low = 0x80;
            high = 0xbf;
        }
    }
    if (valid)
        AddString(text);
    else
        AddNumber(0xfffffffdu);
}

std::size_t TypeSize(int pixel_type)
{
    return pixel_type == TINYEXR_PIXELTYPE_HALF ? 2 : 4;
}

// The values of one channel, `w` by `h` of them, of one of four kinds: noise, a ramp, one
// value, values such as a picture has.
std::vector<unsigned char> MakeValues(int pixel_type, int w, int h)
{
    const std::uint32_t kind = Next() % 4;
    const std::uint32_t first = Next();
    const std::uint32_t step_x = Next() % 8;
    const std::uint32_t step_y = Next() % 8;
    const std::size_t size = TypeSize(pixel_type);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * size);
    std::size_t at = 0;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            std::uint32_t value;
            switch (kind)
            {
                case 0: value = Next(); break;
                case 1: value = first + static_cast<std::uint32_t>(x) * step_x + static_cast<std::uint32_t>(y) * step_y; break;
                case 2: value = first; break;
                default:
                {
                    const std::uint32_t n = Next();
                    value = pixel_type == TINYEXR_PIXELTYPE_FLOAT ? 0x3c000000u + n % 0x04000000u : pixel_type == TINYEXR_PIXELTYPE_HALF ? 0x3000u + n % 0x1000u : n % 1000u;
                    break;
                }
            }
            bytes[at++] = static_cast<unsigned char>(value);
            bytes[at++] = static_cast<unsigned char>(value >> 8);
            if (size == 4)
            {
                bytes[at++] = static_cast<unsigned char>(value >> 16);
                bytes[at++] = static_cast<unsigned char>(value >> 24);
            }
        }
    }
    return bytes;
}

struct Channel
{
    std::string name;
    int memory_type; // the type of the values handed to the writer
    int file_type;   // the type they are stored as
};

struct Attribute
{
    std::string name, type;
    std::vector<unsigned char> value;
};

struct Level
{
    int level_x, level_y, width, height, tiles_x, tiles_y;
    std::vector<std::vector<std::vector<unsigned char>>> tiles; // of each tile, of each channel
};

// A synthetic image and how to save it.
struct Made
{
    int width = 0, height = 0;
    bool tiled = false;
    int tile_size_x = 0, tile_size_y = 0, level_mode = 0, rounding_mode = 0;
    bool without_last_level = false;
    int compression = 0;
    int long_name = 0;
    std::vector<Channel> channels;
    std::vector<std::vector<unsigned char>> images; // of each channel, for scanlines
    std::vector<Level> levels;
    std::vector<Attribute> attributes;
    // Whether the count of its chunks is to be among its attributes, as it is or wrong.
    bool with_chunk_count = false;
    std::uint32_t wrong_chunk_count = 0;
    // Whether tinyexr has to read what it saves of it: not with attributes that say other
    // than the image is.
    bool plain = true;
};

const char* const kNames[8] = { "A", "B", "G", "R", "Z", "id", "light.B", "light.G" };

// The count of its chunks into the attribute for it, if it has one: the count, or another.
void SetChunkCount(Made& m, std::uint32_t count)
{
    for (Attribute& attribute : m.attributes)
    {
        if (attribute.name == "chunkCount")
        {
            std::uint32_t value = count;
            if (m.wrong_chunk_count)
            {
                const std::uint32_t other = m.wrong_chunk_count - 1;
                value = other >= count ? other + 1 : other;
            }
            attribute.value[0] = static_cast<unsigned char>(value);
            attribute.value[1] = static_cast<unsigned char>(value >> 8);
            attribute.value[2] = static_cast<unsigned char>(value >> 16);
            attribute.value[3] = static_cast<unsigned char>(value >> 24);
            if (value != count)
                m.plain = false;
        }
    }
}

// Whether the images are made for ZFP compression (the `tinyexr_zfp` dumps): of floats, most
// of them of whole blocks of four by four, with the attributes of ZFP's settings.
bool g_zfp = false;

// A double such as the settings of ZFP have: a rate, a tolerance, or one that is not either.
// (A rate of less than nothing asks for most of four gigabits a block, which is more than a
// gigabyte for two blocks or more, and so is refused; for an image or a tile of one block,
// of one channel, it would be written, as half a gigabyte. `many` says there is more than
// one.)
double ZfpDouble(bool tolerance, bool many)
{
    static const double kRates[16] = { 0.5, 1, 2, 2, 4, 7.3, 8, 12, 16, 24, 31.9, 32, 40, 0, -0.01, 1e-9 };
    static const double kTolerances[16] = { 1e-3, 0.5, 1e-7, 0.01, 2, 1e-30, 0, -1, 1e-40, 1e30, 1e-320, 0.25, 3, 1e-5, 100, 1e300 };
    const std::uint32_t kind = Next() % 32;
    if (kind < 16)
        return tolerance ? kTolerances[kind] : kRates[kind];
    if (kind < 28)
        return static_cast<double>(Next() % 400) / (tolerance ? 1000.0 : 10.0);
    // Not a number, infinity, and (as a rate) more bits than there may be.
    if (kind == 28)
        return std::numeric_limits<double>::quiet_NaN();
    if (kind == 29)
        return std::numeric_limits<double>::infinity();
    return kind == 30 && many ? -1.0 : 100.0;
}

void AddDouble(Made& m, const char* name, double value)
{
    Attribute attribute;
    attribute.name = name;
    attribute.type = "double";
    attribute.value.resize(8);
    std::memcpy(attribute.value.data(), &value, 8);
    m.attributes.push_back(attribute);
}

Made MakeImage(bool small)
{
    Made m;
    m.tiled = Next() % 3 == 0;
    const std::uint32_t size_kind = g_zfp ? 8 + Next() % 8 : small ? 0 : Next() % 8;
    if (size_kind >= 8)
    {
        // (For ZFP: of whole blocks, now and then not.)
        if (size_kind < 15)
        {
            m.width = 4 * (1 + static_cast<int>(Next() % 12));
            m.height = 4 * (1 + static_cast<int>(Next() % 12));
        }
        else
        {
            m.width = 1 + static_cast<int>(Next() % 40);
            m.height = 1 + static_cast<int>(Next() % 40);
        }
    }
    else if (size_kind < 5)
    {
        m.width = 1 + static_cast<int>(Next() % (small ? 10 : 40));
        m.height = 1 + static_cast<int>(Next() % (small ? 36 : 40));
    }
    else if (size_kind < 7)
    {
        m.width = 1 + static_cast<int>(Next() % 150);
        m.height = 1 + static_cast<int>(Next() % 150);
    }
    else
    {
        m.width = 1 + static_cast<int>(Next() % 400);
        m.height = 1 + static_cast<int>(Next() % 100);
    }

    // Its channels: red, green, blue and alpha; without alpha; one channel; or some of eight.
    std::uint32_t count, start;
    const std::uint32_t names_kind = Next() % 4;
    if (names_kind == 0)
    {
        count = 4;
        start = 0;
    }
    else if (names_kind == 1)
    {
        count = 3;
        start = 1;
    }
    else if (names_kind == 2)
    {
        count = 1;
        start = 0;
    }
    else
    {
        count = 1 + Next() % 5;
        start = Next() % (8 - count + 1);
    }
    for (std::uint32_t c = 0; c < count; ++c)
    {
        Channel channel;
        channel.name = names_kind == 2 ? "Y" : kNames[start + c];
        if (g_zfp)
        {
            // Floats, halves or integers handed over, stored as floats, now and then not.
            const std::uint32_t type = Next() % 16;
            channel.memory_type = type < 10 ? TINYEXR_PIXELTYPE_FLOAT : type < 15 ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_UINT;
            const std::uint32_t as_half = Next() % 24 == 0;
            channel.file_type = as_half && channel.memory_type != TINYEXR_PIXELTYPE_UINT ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
            m.channels.push_back(channel);
            continue;
        }
        channel.memory_type = static_cast<int>(Next() % 3);
        const std::uint32_t as_half = Next() % 2;
        channel.file_type = channel.memory_type == TINYEXR_PIXELTYPE_UINT ? TINYEXR_PIXELTYPE_UINT : as_half ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
        m.channels.push_back(channel);
    }

    // Its compression: one of the eight, PIZ more often; now and then one that is not written.
    const std::uint32_t compression = Next() % 20;
    m.compression = compression < 16 ? static_cast<int>(compression % 8) : compression == 16 ? TINYEXR_COMPRESSIONTYPE_ZFP : compression == 17 ? 9 : TINYEXR_COMPRESSIONTYPE_PIZ;
    if (g_zfp)
        m.compression = TINYEXR_COMPRESSIONTYPE_ZFP;
    // (A ZFP file without the attributes of its settings, as these other images are, is
    // written, at 2 bits a value, and not read.)
    else if (m.compression == TINYEXR_COMPRESSIONTYPE_ZFP)
        m.plain = false;
    m.long_name = Next() % 8 == 0;
    if (g_zfp)
    {
        // The settings: a rate, a precision or a tolerance, each with its type; now and then
        // a type it does not know, none, a type of other than one byte, or no value for it.
        const std::uint32_t kind = Next() % 32;
        const std::uint32_t type = kind < 12 ? 0 : kind < 18 ? 1 : kind < 28 ? 2 : kind == 28 ? 3 + Next() % 253 : 0;
        if (kind != 29)
        {
            Attribute attribute;
            attribute.name = "zfpCompressionType";
            attribute.type = "uchar";
            attribute.value.assign(kind == 30 ? 2 : 1, static_cast<unsigned char>(type));
            m.attributes.push_back(attribute);
        }
        if (kind != 31)
        {
            if (type == 0)
                AddDouble(m, "zfpCompressionRate", ZfpDouble(false, m.channels.size() > 1));
            else if (type == 1)
            {
                Attribute attribute;
                attribute.name = "zfpCompressionPrecision";
                attribute.type = "int";
                const std::uint32_t precision = Next() % 70;
                attribute.value.assign(4, 0);
                std::memcpy(attribute.value.data(), &precision, 4);
                m.attributes.push_back(attribute);
            }
            else
                AddDouble(m, "zfpCompressionTolerance", ZfpDouble(true, m.channels.size() > 1));
        }
        // (A file without them saves, at 2 bits a value, and is not read.)
        if (kind >= 28)
            m.plain = false;
    }
    // Now and then attributes of its own: text; a type, a name or a count of chunks, which
    // the parts of a file of many parts have and tinyexr reads in any file; more attributes
    // than tinyexr keeps.
    if (!g_zfp && Next() % 3 == 0)
    {
        const std::uint32_t attributes = 1 + Next() % 3;
        for (std::uint32_t j = 0; j < attributes; ++j)
        {
            const std::uint32_t kind = Next() % 16;
            Attribute attribute;
            attribute.type = "string";
            if (kind < 6)
            {
                attribute.name = kind < 4 ? "comment" : "owner";
                attribute.value.resize(Next() % 20);
                for (unsigned char& byte : attribute.value)
                    byte = static_cast<unsigned char>('a' + Next() % 26);
            }
            else if (kind < 10)
            {
                static const char* const kTypes[5] = { "scanlineimage", "tiledimage", "deeptile", "deepscanline", "other" };
                const char* type = kTypes[Next() % 5];
                attribute.name = "type";
                attribute.value.assign(type, type + std::strlen(type));
                m.plain = false;
            }
            else if (kind < 12)
            {
                attribute.name = "name";
                attribute.value.resize(1 + Next() % 8);
                for (unsigned char& byte : attribute.value)
                    byte = static_cast<unsigned char>('a' + Next() % 26);
            }
            else if (kind < 15)
            {
                // (Its value is filled in once the image is made.)
                const std::uint32_t wrong = Next() % 4;
                const std::uint32_t count = Next() % 40;
                if (m.with_chunk_count)
                    continue;
                m.with_chunk_count = true;
                m.wrong_chunk_count = wrong == 0 ? count + 1 : 0;
                attribute.name = "chunkCount";
                attribute.type = "int";
                attribute.value.resize(4);
            }
            else
            {
                const std::uint32_t many = 126 + Next() % 6;
                attribute.type = "int";
                attribute.value.assign(4, static_cast<unsigned char>(j));
                for (std::uint32_t k = 0; k < many; ++k)
                {
                    char name[8];
                    std::snprintf(name, sizeof name, "a%03u", k);
                    attribute.name = name;
                    m.attributes.push_back(attribute);
                }
                continue;
            }
            m.attributes.push_back(attribute);
        }
    }

    if (!m.tiled)
    {
        for (const Channel& channel : m.channels)
            m.images.push_back(MakeValues(channel.memory_type, m.width, m.height));
        const int lines = tinyexr::NumScanlines(m.compression);
        SetChunkCount(m, static_cast<std::uint32_t>((m.height + lines - 1) / lines));
        return m;
    }

    m.tile_size_x = 1 + static_cast<int>(Next() % static_cast<std::uint32_t>(std::min(m.width, 64)));
    m.tile_size_y = 1 + static_cast<int>(Next() % static_cast<std::uint32_t>(std::min(m.height, 64)));
    if (g_zfp && Next() % 8 != 0)
    {
        // (Of whole blocks, and of one level, most of them.)
        m.tile_size_x = (m.tile_size_x + 3) & ~3;
        m.tile_size_y = (m.tile_size_y + 3) & ~3;
    }
    m.level_mode = g_zfp && Next() % 4 != 0 ? TINYEXR_TILE_ONE_LEVEL : static_cast<int>(Next() % 3);
    m.rounding_mode = static_cast<int>(Next() % 2);
    m.without_last_level = Next() % 16 == 0;

    // Its levels, as tinyexr counts them.
    EXRHeader header;
    InitEXRHeader(&header);
    header.data_window.max_x = m.width - 1;
    header.data_window.max_y = m.height - 1;
    header.tile_size_x = m.tile_size_x;
    header.tile_size_y = m.tile_size_y;
    header.tile_level_mode = m.level_mode;
    header.tile_rounding_mode = m.rounding_mode;
    std::vector<int> num_x_tiles, num_y_tiles;
    if (!tinyexr::PrecalculateTileInfo(num_x_tiles, num_y_tiles, &header))
    {
        std::fprintf(stderr, "tinyexr does not count the tiles of a synthetic image\n");
        std::exit(2);
    }
    const int levels_x = static_cast<int>(num_x_tiles.size());
    const int levels_y = m.level_mode == TINYEXR_TILE_RIPMAP_LEVELS ? static_cast<int>(num_y_tiles.size()) : 1;
    for (int ly = 0; ly < levels_y; ++ly)
    {
        for (int lx = 0; lx < levels_x; ++lx)
        {
            Level level;
            level.level_x = lx;
            level.level_y = m.level_mode == TINYEXR_TILE_RIPMAP_LEVELS ? ly : lx;
            level.width = tinyexr::LevelSize(m.width, level.level_x, m.rounding_mode);
            level.height = tinyexr::LevelSize(m.height, level.level_y, m.rounding_mode);
            level.tiles_x = num_x_tiles[static_cast<std::size_t>(level.level_x)];
            level.tiles_y = num_y_tiles[static_cast<std::size_t>(level.level_y)];
            for (int t = 0; t < level.tiles_x * level.tiles_y; ++t)
            {
                std::vector<std::vector<unsigned char>> tile;
                for (const Channel& channel : m.channels)
                    tile.push_back(MakeValues(channel.memory_type, m.tile_size_x, m.tile_size_y));
                level.tiles.push_back(tile);
            }
            m.levels.push_back(level);
        }
    }
    std::uint32_t chunks = 0;
    for (const Level& level : m.levels)
        chunks += static_cast<std::uint32_t>(level.tiles.size());
    SetChunkCount(m, chunks);
    return m;
}

// A synthetic image as tinyexr's writer takes it: its header, and its first level (which leads to
// the others).
struct Prepared
{
    EXRHeader header;
    std::vector<EXRChannelInfo> channels;
    std::vector<int> pixel_types, requested_pixel_types;
    std::vector<EXRAttribute> attributes;
    std::vector<EXRImage> images;
    std::vector<unsigned char*> pointers;
    std::vector<std::vector<EXRTile>> tiles;
    std::vector<std::vector<std::vector<unsigned char*>>> tile_pointers;

    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    explicit Prepared(Made& m);
};

Prepared::Prepared(Made& m)
{
    const std::size_t count = m.channels.size();
    InitEXRHeader(&header);
    channels.resize(count);
    pixel_types.resize(count);
    requested_pixel_types.resize(count);
    for (std::size_t c = 0; c < count; ++c)
    {
        std::memset(&channels[c], 0, sizeof(EXRChannelInfo));
        std::strncpy(channels[c].name, m.channels[c].name.c_str(), 255);
        pixel_types[c] = m.channels[c].memory_type;
        requested_pixel_types[c] = m.channels[c].file_type;
    }
    header.num_channels = static_cast<int>(count);
    header.channels = channels.data();
    header.pixel_types = pixel_types.data();
    header.requested_pixel_types = requested_pixel_types.data();
    header.compression_type = m.compression;
    header.long_name = m.long_name;
    header.data_window.max_x = m.width - 1;
    header.data_window.max_y = m.height - 1;
    attributes.resize(m.attributes.size());
    for (std::size_t j = 0; j < attributes.size(); ++j)
    {
        std::memset(&attributes[j], 0, sizeof(EXRAttribute));
        std::strncpy(attributes[j].name, m.attributes[j].name.c_str(), 255);
        std::strncpy(attributes[j].type, m.attributes[j].type.c_str(), 255);
        attributes[j].value = m.attributes[j].value.data();
        attributes[j].size = static_cast<int>(m.attributes[j].value.size());
    }
    header.num_custom_attributes = static_cast<int>(attributes.size());
    header.custom_attributes = attributes.data();

    images.resize(m.tiled ? m.levels.size() : 1);
    pointers.resize(count);
    tiles.resize(m.levels.size());
    tile_pointers.resize(m.levels.size());
    for (EXRImage& image : images)
        InitEXRImage(&image);
    if (!m.tiled)
    {
        for (std::size_t c = 0; c < count; ++c)
            pointers[c] = m.images[c].data();
        images[0].images = pointers.data();
        images[0].width = m.width;
        images[0].height = m.height;
        images[0].num_channels = static_cast<int>(count);
    }
    else
    {
        header.tiled = 1;
        header.tile_size_x = m.tile_size_x;
        header.tile_size_y = m.tile_size_y;
        header.tile_level_mode = m.level_mode;
        header.tile_rounding_mode = m.rounding_mode;
        const std::size_t given = m.levels.size() - (m.without_last_level && m.levels.size() > 1 ? 1 : 0);
        for (std::size_t l = 0; l < m.levels.size(); ++l)
        {
            Level& level = m.levels[l];
            tiles[l].resize(level.tiles.size());
            tile_pointers[l].resize(level.tiles.size());
            for (std::size_t t = 0; t < level.tiles.size(); ++t)
            {
                EXRTile& tile = tiles[l][t];
                tile.offset_x = static_cast<int>(t) % level.tiles_x;
                tile.offset_y = static_cast<int>(t) / level.tiles_x;
                tile.level_x = level.level_x;
                tile.level_y = level.level_y;
                tile.width = std::min(m.tile_size_x, level.width - tile.offset_x * m.tile_size_x);
                tile.height = std::min(m.tile_size_y, level.height - tile.offset_y * m.tile_size_y);
                for (std::size_t c = 0; c < count; ++c)
                    tile_pointers[l][t].push_back(level.tiles[t][c].data());
                tile.images = tile_pointers[l][t].data();
            }
            images[l].tiles = tiles[l].data();
            images[l].num_tiles = static_cast<int>(tiles[l].size());
            images[l].level_x = level.level_x;
            images[l].level_y = level.level_y;
            images[l].width = level.width;
            images[l].height = level.height;
            images[l].num_channels = static_cast<int>(count);
            images[l].next_level = l + 1 < given ? &images[l + 1] : NULL;
        }
    }
}

// What the port refuses to write with ZFP compression, with a message of its own (as
// `check` in src/tinyexr/encode.rs has it), or none: a block or a tile that is not of whole
// blocks of four by four, for which tinyexr writes a length it never set; settings that ask
// for more than a gigabyte for a block. (Where a channel is not to be stored as floats,
// tinyexr refuses the image itself, first.)
const char* ZfpRefusal(const Prepared& prepared)
{
    const EXRHeader& header = prepared.header;
    if (header.compression_type != TINYEXR_COMPRESSIONTYPE_ZFP)
        return NULL;
    for (int c = 0; c < header.num_channels; ++c)
    {
        if (header.requested_pixel_types[c] != TINYEXR_PIXELTYPE_FLOAT)
            return NULL;
    }
    tinyexr::ZFPCompressionParam param;
    if (!tinyexr::FindZFPCompressionParam(&param, header.custom_attributes, header.num_custom_attributes, NULL))
    {
        param.type = 0;
        param.rate = 2;
    }
    std::vector<std::pair<int, int>> sizes;
    const EXRImage& image = prepared.images[0];
    if (!image.tiles)
    {
        for (int start = 0; start < image.height; start += 16)
            sizes.push_back(std::make_pair(image.width, std::min(16, image.height - start)));
    }
    else
    {
        for (const EXRImage* level = &image; level; level = level->next_level)
        {
            for (int t = 0; t < level->num_tiles; ++t)
                sizes.push_back(std::make_pair(level->tiles[t].width, level->tiles[t].height));
        }
    }
    for (const std::pair<int, int>& size : sizes)
    {
        if (size.first % 4 != 0 || size.second % 4 != 0)
            return "ZFP compression needs blocks and tiles whose width and height are multiples of four.";
    }
    for (const std::pair<int, int>& size : sizes)
    {
        zfp_stream* zfp = zfp_stream_open(NULL);
        if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_RATE)
            zfp_stream_set_rate(zfp, param.rate, zfp_type_float, 2, 0);
        else if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_PRECISION)
            zfp_stream_set_precision(zfp, param.precision);
        else
            zfp_stream_set_accuracy(zfp, param.tolerance);
        zfp_field* field = zfp_field_2d(NULL, zfp_type_float, static_cast<unsigned int>(size.first), static_cast<unsigned int>(size.second * header.num_channels));
        const std::size_t bytes = zfp_stream_maximum_size(zfp, field);
        zfp_field_free(field);
        zfp_stream_close(zfp);
        if (bytes > (static_cast<std::size_t>(1) << 30))
            return "ZFP compression with these settings needs more than a gigabyte for a block.";
    }
    return NULL;
}

// The image saved by tinyexr: the file's bytes, or none, tinyexr's message, and what it gave
// as the size (nothing, or one of its error numbers).
std::vector<unsigned char> Save(Made& m, std::string& message, std::uint32_t& refusal)
{
    Prepared prepared(m);
    EXRHeader& header = prepared.header;
    std::vector<EXRImage>& images = prepared.images;

    std::vector<unsigned char> file;
    if (const char* refused = ZfpRefusal(prepared))
    {
        refusal = static_cast<std::uint32_t>(TINYEXR_ERROR_INVALID_ARGUMENT);
        message = refused;
        return file;
    }

    unsigned char* memory = NULL;
    const char* err = NULL;
    const std::size_t size = SaveEXRImageToMemory(&images[0], &header, &memory, &err);
    // (Nothing, or one of its error numbers as a size.)
    if (size == 0 || size > (static_cast<std::size_t>(1) << 40))
    {
        refusal = static_cast<std::uint32_t>(size);
        message = err ? err : "";
        FreeEXRErrorMessage(err);
        return file;
    }
    file.assign(memory, memory + size);
    std::free(memory);
    return file;
}

// Where the value of an attribute starts in a file, or 0: after its name, its type and its
// size.
std::size_t FindAttribute(const std::vector<unsigned char>& file, const char* name, const char* type)
{
    std::string key = std::string(name) + '\0' + type + '\0';
    const auto found = std::search(file.begin(), file.end(), key.begin(), key.end());
    if (found == file.end())
        return 0;
    return static_cast<std::size_t>(found - file.begin()) + key.size() + 4;
}

std::uint32_t GetNumber(const std::vector<unsigned char>& file, std::size_t at)
{
    return static_cast<std::uint32_t>(file[at]) | static_cast<std::uint32_t>(file[at + 1]) << 8 | static_cast<std::uint32_t>(file[at + 2]) << 16 | static_cast<std::uint32_t>(file[at + 3]) << 24;
}

// (Nothing where the number would not be whole within the file.)
void SetNumber(std::vector<unsigned char>& file, std::size_t at, std::uint32_t value)
{
    if (at + 4 > file.size())
        return;
    file[at] = static_cast<unsigned char>(value);
    file[at + 1] = static_cast<unsigned char>(value >> 8);
    file[at + 2] = static_cast<unsigned char>(value >> 16);
    file[at + 3] = static_cast<unsigned char>(value >> 24);
}

void AddToNumber(std::vector<unsigned char>& file, std::size_t at, std::uint32_t by)
{
    if (at + 4 <= file.size())
        SetNumber(file, at, GetNumber(file, at) + by);
}

// Where the header of a file tinyexr saved ends, which is where its table of offsets starts;
// 0 where it does not end.
std::size_t EndOfHeader(const std::vector<unsigned char>& file)
{
    std::size_t at = 8;
    while (at < file.size() && file[at])
    {
        for (int text = 0; text < 2; ++text)
        {
            while (at < file.size() && file[at])
                ++at;
            ++at;
        }
        if (at + 4 > file.size())
            return 0;
        at += 4 + static_cast<std::size_t>(GetNumber(file, at));
    }
    return at < file.size() ? at + 1 : 0;
}

// A number such as a damaged header has: nothing, less than nothing, the most and the least
// there are, the most tinyexr takes for a width and one more, small ones of either sign.
std::uint32_t Special()
{
    const std::uint32_t kind = Next() % 8;
    const std::uint32_t small = Next() % 64;
    switch (kind)
    {
        case 0: return 0;
        case 1: return 0xffffffffu;
        case 2: return 0x7fffffffu;
        case 3: return 0x80000000u;
        case 4: return 1024 * 8192;
        case 5: return 1024 * 8192 + 1;
        case 6: return small;
        default: return 0u - small;
    }
}

#ifdef SHOW_EACH
#define STAGE(name) std::fprintf(stderr, "%s %016llx\n", name, static_cast<unsigned long long>(g_hash))
#else
#define STAGE(name)
#endif

void AddMessage(const char* err)
{
    if (err)
    {
#ifdef SHOW_REASONS
        std::fprintf(stderr, "%s\n", err);
#endif
        AddString(err);
        FreeEXRErrorMessage(err);
    }
    else
        AddNumber(0xfffffffeu);
}

struct Counts
{
    std::uint32_t headers = 0, images = 0, refused = 0, left = 0, as_rgba = 0, stopped = 0;
};

// What tinyexr reads of a file: its version, its header, its image, and it as floats of red,
// green, blue and alpha. Whether the image was read.
//
// tinyexr reads the lines of a block that is not compressed, for a channel of halves read as
// halves, without a look at the block's length, and so past the end of the file where a
// damaged block is its last. The port refuses such a block. So that the two can be compared, a
// damaged file that is not compressed is read with halves as floats, where tinyexr looks.
//
// And tinyexr throws where a damaged tile's place is past the image's edge, so that the tile
// has no width or no height: it asks a buffer of no bytes for its first. Nothing in it
// catches that, which stops a program. Here it is caught, and `stops` says which of the two
// readings it stopped (4 the image, 8 the image as red, green, blue and alpha), for the file
// of files to tell the port's side, which has to refuse such a file.
bool Read(const unsigned char* bytes, std::size_t size, std::uint32_t wish, Counts& counts, std::uint32_t& stops)
{
    bool read = false;
    bool large = false;
    EXRVersion version;
    int ret = ParseEXRVersionFromMemory(&version, bytes, size);
    AddNumber(static_cast<std::uint32_t>(ret));
    if (ret == TINYEXR_SUCCESS)
    {
        AddNumber(static_cast<std::uint32_t>(version.version));
        AddNumber(static_cast<std::uint32_t>(version.tiled));
        AddNumber(static_cast<std::uint32_t>(version.long_name));
        AddNumber(static_cast<std::uint32_t>(version.non_image));
        AddNumber(static_cast<std::uint32_t>(version.multipart));

        EXRHeader header;
        InitEXRHeader(&header);
        const char* err = NULL;
        ret = ParseEXRHeaderFromMemory(&header, &version, bytes, size, &err);
        AddNumber(static_cast<std::uint32_t>(ret));
        AddMessage(err);
        if (ret == TINYEXR_SUCCESS)
        {
            ++counts.headers;
            AddFloat(header.pixel_aspect_ratio);
            AddNumber(static_cast<std::uint32_t>(header.line_order));
            AddNumber(static_cast<std::uint32_t>(header.data_window.min_x));
            AddNumber(static_cast<std::uint32_t>(header.data_window.min_y));
            AddNumber(static_cast<std::uint32_t>(header.data_window.max_x));
            AddNumber(static_cast<std::uint32_t>(header.data_window.max_y));
            AddNumber(static_cast<std::uint32_t>(header.display_window.min_x));
            AddNumber(static_cast<std::uint32_t>(header.display_window.min_y));
            AddNumber(static_cast<std::uint32_t>(header.display_window.max_x));
            AddNumber(static_cast<std::uint32_t>(header.display_window.max_y));
            AddFloat(header.screen_window_center[0]);
            AddFloat(header.screen_window_center[1]);
            AddFloat(header.screen_window_width);
            AddNumber(static_cast<std::uint32_t>(header.chunk_count));
            AddNumber(static_cast<std::uint32_t>(header.tiled));
            AddNumber(static_cast<std::uint32_t>(header.tile_size_x));
            AddNumber(static_cast<std::uint32_t>(header.tile_size_y));
            AddNumber(static_cast<std::uint32_t>(header.tile_level_mode));
            AddNumber(static_cast<std::uint32_t>(header.tile_rounding_mode));
            AddNumber(static_cast<std::uint32_t>(header.long_name));
            AddNumber(static_cast<std::uint32_t>(header.non_image));
            AddNumber(static_cast<std::uint32_t>(header.multipart));
            AddNumber(header.header_len);
            AddNumber(static_cast<std::uint32_t>(header.num_custom_attributes));
            for (int j = 0; j < header.num_custom_attributes; ++j)
            {
                AddName(header.custom_attributes[j].name);
                AddName(header.custom_attributes[j].type);
                AddNumber(static_cast<std::uint32_t>(header.custom_attributes[j].size));
                Add(header.custom_attributes[j].value, static_cast<std::size_t>(header.custom_attributes[j].size));
            }
            AddNumber(static_cast<std::uint32_t>(header.num_channels));
            for (int c = 0; c < header.num_channels; ++c)
            {
                AddName(header.channels[c].name);
                AddNumber(static_cast<std::uint32_t>(header.channels[c].pixel_type));
                AddNumber(static_cast<std::uint32_t>(header.channels[c].x_sampling));
                AddNumber(static_cast<std::uint32_t>(header.channels[c].y_sampling));
                AddNumber(header.channels[c].p_linear);
                AddNumber(static_cast<std::uint32_t>(header.pixel_types[c]));
                AddNumber(static_cast<std::uint32_t>(header.requested_pixel_types[c]));
            }
            AddNumber(static_cast<std::uint32_t>(header.compression_type));
            AddName(header.name);
            STAGE("header");

            // A damaged size can ask for more memory and time than there is; those are left,
            // but for the sizes tinyexr refuses at once. Two more are left, which stop
            // tinyexr: a tile size of 0, which it divides by, and a count of chunks in the
            // thousands of millions, which it makes a table for before it looks whether the
            // file has one.
            {
                const std::int64_t limit = 1024 * 8192;
                const std::int64_t w = static_cast<std::int64_t>(header.data_window.max_x) - header.data_window.min_x + 1;
                const std::int64_t h = static_cast<std::int64_t>(header.data_window.max_y) - header.data_window.min_y + 1;
                const std::int64_t tile_x = header.tile_size_x, tile_y = header.tile_size_y;
                const bool refused = w <= 0 || h <= 0 || w > limit || h > limit || (header.tiled && (tile_x > limit || tile_y > limit));
                if (refused)
                    large = false;
                else if (header.tiled)
                    large = w * h > (1 << 22) || tile_x <= 0 || tile_y <= 0 || tile_x * tile_y > (1 << 16) || (w / tile_x + 1) * (h / tile_y + 1) * tile_x * tile_y > (1 << 23);
                else
                    large = w * h > (1 << 22) || header.chunk_count > (1 << 24);
            }
            if (large)
                ++counts.left;
            else
            {
                if ((wish & 1) || ((wish & 2) && header.compression_type == TINYEXR_COMPRESSIONTYPE_NONE))
                {
                    for (int c = 0; c < header.num_channels; ++c)
                    {
                        if (header.pixel_types[c] == TINYEXR_PIXELTYPE_HALF)
                            header.requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
                    }
                }
                EXRImage image;
                InitEXRImage(&image);
                err = NULL;
                bool threw = false;
                try
                {
                    ret = LoadEXRImageFromMemory(&image, &header, bytes, size, &err);
                }
                catch (const std::exception&)
                {
                    threw = true;
                }
                if (threw)
                {
                    AddNumber(0xfffffffcu);
                    stops |= 4;
                    ++counts.stopped;
                }
                else
                {
                    AddNumber(static_cast<std::uint32_t>(ret));
                    AddMessage(err);
                }
                if (threw)
                    ;
                else if (ret == TINYEXR_SUCCESS)
                {
                    read = true;
                    ++counts.images;
                    for (int c = 0; c < header.num_channels; ++c)
                        AddNumber(static_cast<std::uint32_t>(header.pixel_types[c]));
                    for (const EXRImage* level = &image; level; level = level->next_level)
                    {
                        AddNumber(static_cast<std::uint32_t>(level->level_x));
                        AddNumber(static_cast<std::uint32_t>(level->level_y));
                        AddNumber(static_cast<std::uint32_t>(level->width));
                        AddNumber(static_cast<std::uint32_t>(level->height));
                        AddNumber(static_cast<std::uint32_t>(level->num_channels));
                        AddNumber(static_cast<std::uint32_t>(level->num_tiles));
                        if (level->tiles)
                        {
                            for (int t = 0; t < level->num_tiles; ++t)
                            {
                                const EXRTile& tile = level->tiles[t];
                                AddNumber(static_cast<std::uint32_t>(tile.offset_x));
                                AddNumber(static_cast<std::uint32_t>(tile.offset_y));
                                AddNumber(static_cast<std::uint32_t>(tile.level_x));
                                AddNumber(static_cast<std::uint32_t>(tile.level_y));
                                AddNumber(static_cast<std::uint32_t>(tile.width));
                                AddNumber(static_cast<std::uint32_t>(tile.height));
                                for (int c = 0; c < level->num_channels; ++c)
                                    Add(tile.images[c], static_cast<std::size_t>(header.tile_size_x) * static_cast<std::size_t>(header.tile_size_y) * TypeSize(header.requested_pixel_types[c]));
                            }
                        }
                        else
                        {
                            for (int c = 0; c < level->num_channels; ++c)
                                Add(level->images[c], static_cast<std::size_t>(level->width) * static_cast<std::size_t>(level->height) * TypeSize(header.requested_pixel_types[c]));
                        }
                    }
                    FreeEXRImage(&image);
                }
                else
                    ++counts.refused;
                STAGE("image");
            }
        }
        FreeEXRHeader(&header);
    }

    if (!large)
    {
        float* rgba = NULL;
        int width = 0, height = 0;
        const char* err = NULL;
        bool threw = false;
        try
        {
            ret = LoadEXRFromMemory(&rgba, &width, &height, bytes, size, &err);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        if (threw)
        {
            AddNumber(0xfffffffcu);
            stops |= 8;
        }
        else
        {
            AddNumber(static_cast<std::uint32_t>(ret));
            AddMessage(err);
        }
        if (!threw && ret == TINYEXR_SUCCESS)
        {
            ++counts.as_rgba;
            AddNumber(static_cast<std::uint32_t>(width));
            AddNumber(static_cast<std::uint32_t>(height));
            Add(rgba, static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 16);
            std::free(rgba);
        }
    }
    return read;
}

void WriteNumber(std::FILE* file, std::uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = static_cast<unsigned char>(value);
    bytes[1] = static_cast<unsigned char>(value >> 8);
    bytes[2] = static_cast<unsigned char>(value >> 16);
    bytes[3] = static_cast<unsigned char>(value >> 24);
    std::fwrite(bytes, 1, 4, file);
}

// The `tinyexr_write` dumps.
int Write(const char* name, bool small, std::uint32_t count)
{
    std::uint64_t file_total = 0;
    std::uint32_t saved = 0, not_saved = 0, short_way = 0;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        Made made = MakeImage(small);
        std::string message;
        std::uint32_t refusal = 0;
        const std::vector<unsigned char> file = Save(made, message, refusal);
        if (file.empty())
        {
            ++not_saved;
            AddNumber(refusal);
            AddString(message.c_str());
        }
        else
        {
            ++saved;
            file_total += file.size();
            AddNumber(static_cast<std::uint32_t>(file.size()));
            Add(file.data(), file.size());
        }

        // Every fourth, floats saved the short way: one, three or four to a pixel (or two,
        // which it does not take), as halves or as floats.
        if (i % 4 == 0)
        {
            static const int kComponents[8] = { 2, 1, 1, 3, 3, 3, 4, 4 };
            const int components = kComponents[Next() % 8];
            const int width = 1 + static_cast<int>(Next() % 40);
            const int height = 1 + static_cast<int>(Next() % 40);
            const int as_halves = static_cast<int>(Next() % 2);
            const std::vector<unsigned char> bytes = MakeValues(TINYEXR_PIXELTYPE_FLOAT, width * components, height);
            std::vector<float> floats(bytes.size() / 4);
            std::memcpy(floats.data(), bytes.data(), bytes.size());
            unsigned char* memory = NULL;
            const char* err = NULL;
            const int size = SaveEXRToMemory(floats.data(), width, height, components, as_halves, &memory, &err);
            AddNumber(static_cast<std::uint32_t>(size));
            if (size > 0)
            {
                ++short_way;
                file_total += static_cast<std::size_t>(size);
                Add(memory, static_cast<std::size_t>(size));
                std::free(memory);
            }
            else
            {
                AddString(err ? err : "");
                FreeEXRErrorMessage(err);
            }
        }

#ifdef SHOW_EACH
        std::fprintf(stderr, "file %u %016llx\n", i, static_cast<unsigned long long>(g_hash));
#endif
        if ((i + 1) % 200 == 0 || i + 1 == count)
            std::printf("%s %u %016llx\n", name, i + 1, static_cast<unsigned long long>(g_hash));
    }
    std::printf("%s bytes of files: %llu; saved: %u, not saved: %u, saved the short way: %u\n", name, static_cast<unsigned long long>(file_total), saved, not_saved, short_way);
    return 0;
}

// ---------- Files of many parts, and deep images ----------

// A second generator, for what is done to the files of many parts: so that the first makes
// the same images in both programs.
std::uint64_t g_state2;
std::uint32_t Next2()
{
    g_state2 = g_state2 * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(g_state2 >> 32);
}

// `Special`, from the second generator.
std::uint32_t Special2()
{
    const std::uint32_t kind = Next2() % 8;
    const std::uint32_t small = Next2() % 64;
    static const std::uint32_t kValues[6] = { 0, 0xffffffffu, 0x7fffffffu, 0x80000000u, 1024 * 8192, 1024 * 8192 + 1 };
    return kind < 6 ? kValues[kind] : kind == 6 ? small : 0u - small;
}

// A message, or a mark where there is none or it is empty.
void AddMessage2(const char* err)
{
    if (err && err[0])
        AddString(err);
    else
        AddNumber(0xfffffffeu);
    if (err)
        FreeEXRErrorMessage(err);
}

void AddHeader(const EXRHeader& header)
{
    AddFloat(header.pixel_aspect_ratio);
    AddNumber(static_cast<std::uint32_t>(header.line_order));
    AddNumber(static_cast<std::uint32_t>(header.data_window.min_x));
    AddNumber(static_cast<std::uint32_t>(header.data_window.min_y));
    AddNumber(static_cast<std::uint32_t>(header.data_window.max_x));
    AddNumber(static_cast<std::uint32_t>(header.data_window.max_y));
    AddNumber(static_cast<std::uint32_t>(header.display_window.min_x));
    AddNumber(static_cast<std::uint32_t>(header.display_window.min_y));
    AddNumber(static_cast<std::uint32_t>(header.display_window.max_x));
    AddNumber(static_cast<std::uint32_t>(header.display_window.max_y));
    AddFloat(header.screen_window_center[0]);
    AddFloat(header.screen_window_center[1]);
    AddFloat(header.screen_window_width);
    AddNumber(static_cast<std::uint32_t>(header.chunk_count));
    AddNumber(static_cast<std::uint32_t>(header.tiled));
    AddNumber(static_cast<std::uint32_t>(header.tile_size_x));
    AddNumber(static_cast<std::uint32_t>(header.tile_size_y));
    AddNumber(static_cast<std::uint32_t>(header.tile_level_mode));
    AddNumber(static_cast<std::uint32_t>(header.tile_rounding_mode));
    AddNumber(static_cast<std::uint32_t>(header.long_name));
    AddNumber(static_cast<std::uint32_t>(header.non_image));
    AddNumber(static_cast<std::uint32_t>(header.multipart));
    AddNumber(header.header_len);
    AddNumber(static_cast<std::uint32_t>(header.num_custom_attributes));
    for (int j = 0; j < header.num_custom_attributes; ++j)
    {
        AddName(header.custom_attributes[j].name);
        AddName(header.custom_attributes[j].type);
        AddNumber(static_cast<std::uint32_t>(header.custom_attributes[j].size));
        Add(header.custom_attributes[j].value, static_cast<std::size_t>(header.custom_attributes[j].size));
    }
    AddNumber(static_cast<std::uint32_t>(header.num_channels));
    for (int c = 0; c < header.num_channels; ++c)
    {
        AddName(header.channels[c].name);
        AddNumber(static_cast<std::uint32_t>(header.channels[c].pixel_type));
        AddNumber(static_cast<std::uint32_t>(header.channels[c].x_sampling));
        AddNumber(static_cast<std::uint32_t>(header.channels[c].y_sampling));
        AddNumber(header.channels[c].p_linear);
        AddNumber(static_cast<std::uint32_t>(header.pixel_types[c]));
        AddNumber(static_cast<std::uint32_t>(header.requested_pixel_types[c]));
    }
    AddNumber(static_cast<std::uint32_t>(header.compression_type));
    AddName(header.name);
}

void AddImage(const EXRImage& image, const EXRHeader& header)
{
    for (int c = 0; c < header.num_channels; ++c)
        AddNumber(static_cast<std::uint32_t>(header.pixel_types[c]));
    for (const EXRImage* level = &image; level; level = level->next_level)
    {
        AddNumber(static_cast<std::uint32_t>(level->level_x));
        AddNumber(static_cast<std::uint32_t>(level->level_y));
        AddNumber(static_cast<std::uint32_t>(level->width));
        AddNumber(static_cast<std::uint32_t>(level->height));
        AddNumber(static_cast<std::uint32_t>(level->num_channels));
        AddNumber(static_cast<std::uint32_t>(level->num_tiles));
        if (level->tiles)
        {
            for (int t = 0; t < level->num_tiles; ++t)
            {
                const EXRTile& tile = level->tiles[t];
                AddNumber(static_cast<std::uint32_t>(tile.offset_x));
                AddNumber(static_cast<std::uint32_t>(tile.offset_y));
                AddNumber(static_cast<std::uint32_t>(tile.level_x));
                AddNumber(static_cast<std::uint32_t>(tile.level_y));
                AddNumber(static_cast<std::uint32_t>(tile.width));
                AddNumber(static_cast<std::uint32_t>(tile.height));
                for (int c = 0; c < level->num_channels; ++c)
                    Add(tile.images[c], static_cast<std::size_t>(header.tile_size_x) * static_cast<std::size_t>(header.tile_size_y) * TypeSize(header.requested_pixel_types[c]));
            }
        }
        else
        {
            for (int c = 0; c < level->num_channels; ++c)
                Add(level->images[c], static_cast<std::size_t>(level->width) * static_cast<std::size_t>(level->height) * TypeSize(header.requested_pixel_types[c]));
        }
    }
}

// Whether a part is left unread: by the rule for a file of one part.
bool Left(const EXRHeader& header)
{
    const std::int64_t limit = 1024 * 8192;
    const std::int64_t w = static_cast<std::int64_t>(header.data_window.max_x) - header.data_window.min_x + 1;
    const std::int64_t h = static_cast<std::int64_t>(header.data_window.max_y) - header.data_window.min_y + 1;
    const std::int64_t tile_x = header.tile_size_x, tile_y = header.tile_size_y;
    const bool refused = w <= 0 || h <= 0 || w > limit || h > limit || (header.tiled && (tile_x > limit || tile_y > limit));
    if (refused)
        return false;
    if (header.tiled)
        return w * h > (1 << 22) || tile_x <= 0 || tile_y <= 0 || tile_x * tile_y > (1 << 16) || (w / tile_x + 1) * (h / tile_y + 1) * tile_x * tile_y > (1 << 23);
    return w * h > (1 << 22) || header.chunk_count > (1 << 24);
}

bool ReadAll(int fd, void* data, std::size_t size)
{
    unsigned char* at = static_cast<unsigned char*>(data);
    while (size)
    {
        const ssize_t got = read(fd, at, size);
        if (got <= 0)
            return false;
        at += got;
        size -= static_cast<std::size_t>(got);
    }
    return true;
}

// What a reading of a file comes to with zeros outside it and with ones, and whether it
// stopped either time (threw, ended the program, or took too long): in a program of its own.
struct Outcome
{
    std::uint64_t hash[2];
    std::uint32_t stopped;
};

template <typename F>
Outcome Isolated(F reading)
{
    Outcome outcome;
    std::memset(&outcome, 0, sizeof outcome);
    outcome.stopped = 1;
    std::fflush(stdout);
    int fds[2];
    if (pipe(fds) != 0)
        std::exit(2);
    const pid_t pid = fork();
    if (pid < 0)
        std::exit(2);
    if (pid == 0)
    {
        close(fds[0]);
        alarm(20);
        Outcome mine;
        std::memset(&mine, 0, sizeof mine);
        for (int k = 0; k < 2; ++k)
        {
            g_hash = 14695981039346656037ull;
            bool threw = false;
            try
            {
                reading(static_cast<unsigned char>(k ? 0xff : 0x00));
            }
            catch (...)
            {
                threw = true;
            }
            if (threw)
            {
                mine.stopped = 1;
                break;
            }
            mine.hash[k] = g_hash;
        }
        if (write(fds[1], &mine, sizeof mine) != static_cast<ssize_t>(sizeof mine))
            _exit(1);
        _exit(0);
    }
    close(fds[1]);
    Outcome got;
    if (ReadAll(fds[0], &got, sizeof got))
        outcome = got;
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return outcome;
}

// Into the hash, what came of a reading: 1 where it depends on what is outside the file, 2
// where tinyexr stopped, or 0 and the hash of what it read. The flags.
std::uint32_t AddOutcome(const Outcome& outcome)
{
    std::uint32_t flags = 0;
    if (outcome.stopped)
        flags = 2;
    else if (outcome.hash[0] != outcome.hash[1])
        flags = 1;
    AddNumber(flags);
    if (!flags)
        Add(&outcome.hash[0], 8);
    return flags;
}

// The reading of a file of many parts, as examples/dump.rs does it: its version, its headers,
// and the images of its parts (of the parts the wish has a bit for, halves as floats); with
// `pad` past its end.
void ReadParts(const std::vector<unsigned char>& file, std::uint32_t wish, unsigned char pad)
{
    const std::size_t size = file.size();
    std::vector<unsigned char> memory(2 * size + 64, pad);
    std::copy(file.begin(), file.end(), memory.begin());
    EXRVersion version;
    int ret = ParseEXRVersionFromMemory(&version, memory.data(), size);
    AddNumber(static_cast<std::uint32_t>(ret));
    if (ret != TINYEXR_SUCCESS)
        return;
    AddNumber(static_cast<std::uint32_t>(version.version));
    AddNumber(static_cast<std::uint32_t>(version.tiled));
    AddNumber(static_cast<std::uint32_t>(version.long_name));
    AddNumber(static_cast<std::uint32_t>(version.non_image));
    AddNumber(static_cast<std::uint32_t>(version.multipart));

    EXRHeader** headers = NULL;
    int num = 0;
    const char* err = NULL;
    ret = ParseEXRMultipartHeaderFromMemory(&headers, &num, &version, memory.data(), size, &err);
    AddNumber(static_cast<std::uint32_t>(ret));
    AddMessage2(err);
    if (ret != TINYEXR_SUCCESS)
        return;
    AddNumber(static_cast<std::uint32_t>(num));
    bool left = num > 64;
    std::int64_t chunks = 0;
    for (int i = 0; i < num; ++i)
    {
        AddHeader(*headers[i]);
        left = left || Left(*headers[i]);
        chunks += headers[i]->chunk_count > 0 ? headers[i]->chunk_count : 0;
    }
    if (left || chunks > (1 << 16))
    {
        AddNumber(0xfffffffbu);
        return;
    }
    for (int i = 0; i < num; ++i)
    {
        if (i < 32 && ((wish >> i) & 1))
        {
            for (int c = 0; c < headers[i]->num_channels; ++c)
            {
                if (headers[i]->pixel_types[c] == TINYEXR_PIXELTYPE_HALF)
                    headers[i]->requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
            }
        }
    }
    // (Room past the end for the tables of offsets of every part.)
    memory.resize(2 * size + 64 + static_cast<std::size_t>(chunks) * 8, pad);
    std::vector<EXRImage> images(static_cast<std::size_t>(num));
    for (EXRImage& image : images)
        InitEXRImage(&image);
    err = NULL;
    ret = LoadEXRMultipartImageFromMemory(images.data(), const_cast<const EXRHeader**>(headers), static_cast<unsigned int>(num), memory.data(), size, &err);
    AddNumber(static_cast<std::uint32_t>(ret));
    AddMessage2(err);
    if (ret == TINYEXR_SUCCESS)
    {
        for (int i = 0; i < num; ++i)
            AddImage(images[static_cast<std::size_t>(i)], *headers[i]);
    }
}

// The `tinyexr_parts` dump: files of one to four parts (one now and then), each made as for
// the `tinyexr_write_small` dump, saved by tinyexr: the bytes, or the message. Each file (a
// third of them damaged, by the second generator) is read in a program of its own, with zeros
// and with ones past its end, and written to the file of files with what came of it.
int Parts(const char* name, std::uint32_t count, std::FILE* out)
{
    std::uint64_t file_total = 0;
    std::uint32_t saved = 0, not_saved = 0, past_end = 0, stopped = 0;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const std::uint32_t num_parts = Next() % 10 == 0 ? 1 : 2 + Next() % 3;
        std::vector<Made> parts;
        std::vector<std::string> names;
        for (std::uint32_t p = 0; p < num_parts; ++p)
        {
            parts.push_back(MakeImage(true));
            // Its name: its own, none, or the one before's.
            const std::uint32_t kind = Next() % 40;
            names.push_back(kind == 0 ? std::string() : kind == 1 && p > 0 ? names.back() : "part" + std::to_string(p));
        }
        std::vector<unsigned char> file;
        std::string message;
        std::uint32_t refusal = 0;
        {
            std::vector<std::unique_ptr<Prepared>> prepared;
            std::vector<EXRImage> images;
            std::vector<const EXRHeader*> headers;
            for (std::size_t p = 0; p < parts.size(); ++p)
            {
                prepared.emplace_back(new Prepared(parts[p]));
                std::strncpy(prepared.back()->header.name, names[p].c_str(), 255);
                images.push_back(prepared.back()->images[0]);
                headers.push_back(&prepared.back()->header);
            }
            // (What the port refuses of the first part it refuses, unless tinyexr refuses one
            // of them first, for a channel that is not to be stored as floats.)
            const char* refused = NULL;
            bool not_floats = false;
            for (const std::unique_ptr<Prepared>& part : prepared)
            {
                if (part->header.compression_type != TINYEXR_COMPRESSIONTYPE_ZFP)
                    continue;
                for (int c = 0; c < part->header.num_channels; ++c)
                    not_floats = not_floats || part->header.requested_pixel_types[c] != TINYEXR_PIXELTYPE_FLOAT;
                if (!refused)
                    refused = ZfpRefusal(*part);
            }
            unsigned char* memory = NULL;
            const char* err = NULL;
            const std::size_t size = refused && !not_floats ? 0 : SaveEXRMultipartImageToMemory(images.data(), headers.data(), num_parts, &memory, &err);
            if (refused && !not_floats)
            {
                refusal = static_cast<std::uint32_t>(TINYEXR_ERROR_INVALID_ARGUMENT);
                message = refused;
            }
            else if (size == 0 || size > (static_cast<std::size_t>(1) << 40))
            {
                refusal = static_cast<std::uint32_t>(size);
                message = err ? err : "";
                FreeEXRErrorMessage(err);
            }
            else
            {
                file.assign(memory, memory + size);
                std::free(memory);
            }
        }
        if (file.empty())
        {
            ++not_saved;
            AddNumber(refusal);
            AddString(message.c_str());
        }
        else
        {
            ++saved;
            file_total += file.size();
            AddNumber(static_cast<std::uint32_t>(file.size()));
            Add(file.data(), file.size());
        }

        // A third damaged: bits turned over, the end cut off, a byte randomised, a byte of the
        // headers made zero, an offset or the part number of a block randomised.
        const std::uint32_t wish = Next2();
        if (!file.empty() && Next2() % 3 == 0)
        {
            const std::uint32_t how = Next2() % 6;
            if (how == 0)
            {
                for (std::uint32_t k = 1 + Next2() % 3; k > 0; --k)
                {
                    const std::uint32_t bit = Next2() % static_cast<std::uint32_t>(file.size() * 8);
                    file[bit >> 3] ^= static_cast<unsigned char>(1u << (bit & 7));
                }
            }
            else if (how == 1)
                file.resize(Next2() % file.size());
            else if (how == 2)
                file[Next2() % file.size()] = static_cast<unsigned char>(Next2());
            else if (how == 3)
                file[Next2() % std::min<std::size_t>(file.size(), 600)] = 0;
            else
            {
                // Past the headers (each to a zero byte, and a zero after the last), the table
                // of offsets.
                std::size_t at = 8;
                while (at < file.size() && file[at])
                {
                    while (at < file.size() && file[at])
                    {
                        for (int text = 0; text < 2; ++text)
                        {
                            while (at < file.size() && file[at])
                                ++at;
                            ++at;
                        }
                        if (at + 4 > file.size())
                            break;
                        at += 4 + GetNumber(file, at);
                    }
                    ++at;
                }
                ++at;
                const std::size_t entry = at + 8 * (Next2() % 8);
                const std::uint32_t value = Special2();
                if (how == 4)
                    SetNumber(file, entry + 4 * (Next2() % 2), value);
                else if (entry + 8 <= file.size())
                    SetNumber(file, GetNumber(file, entry), value % 5);
            }
        }

        const Outcome outcome = Isolated([&](unsigned char pad) { ReadParts(file, wish, pad); });
        const std::uint32_t flags = AddOutcome(outcome);
        past_end += flags == 1;
        stopped += flags == 2;
        WriteNumber(out, wish);
        WriteNumber(out, flags);
        WriteNumber(out, static_cast<std::uint32_t>(file.size()));
        if (!file.empty())
            std::fwrite(file.data(), 1, file.size(), out);

#ifdef SHOW_EACH
        std::fprintf(stderr, "file %u %016llx\n", i, static_cast<unsigned long long>(g_hash));
#endif
        if ((i + 1) % 200 == 0 || i + 1 == count)
            std::printf("%s %u %016llx\n", name, i + 1, static_cast<unsigned long long>(g_hash));
    }
    std::printf("%s bytes of files: %llu; saved: %u, not saved: %u; read where it depends on what is past the end: %u, where tinyexr stops: %u\n", name, static_cast<unsigned long long>(file_total), saved, not_saved, past_end, stopped);
    return 0;
}

void Put32(std::vector<unsigned char>& f, std::uint32_t v)
{
    for (int k = 0; k < 4; ++k)
        f.push_back(static_cast<unsigned char>(v >> (8 * k)));
}
void Put64(std::vector<unsigned char>& f, std::uint64_t v)
{
    for (int k = 0; k < 8; ++k)
        f.push_back(static_cast<unsigned char>(v >> (8 * k)));
}
void PutAttribute(std::vector<unsigned char>& f, const char* name, const char* type, const std::vector<unsigned char>& data)
{
    f.insert(f.end(), name, name + std::strlen(name) + 1);
    f.insert(f.end(), type, type + std::strlen(type) + 1);
    Put32(f, static_cast<std::uint32_t>(data.size()));
    f.insert(f.end(), data.begin(), data.end());
}

// A block as a deep file stores it: compressed, or now and then as it is.
std::vector<unsigned char> Pack(int compression, const std::vector<unsigned char>& raw)
{
    if (raw.empty() || compression == TINYEXR_COMPRESSIONTYPE_NONE || Next() % 8 == 0)
        return raw;
    std::vector<unsigned char> packed(raw.size() * 2 + 1024);
    tinyexr::tinyexr_uint64 size = 0;
    if (compression == TINYEXR_COMPRESSIONTYPE_RLE)
        tinyexr::CompressRle(packed.data(), size, raw.data(), static_cast<unsigned long>(raw.size()));
    else
        tinyexr::CompressZip(packed.data(), size, raw.data(), static_cast<unsigned long>(raw.size()));
    packed.resize(static_cast<std::size_t>(size));
    return packed;
}

// A synthetic deep image (tinyexr reads them, and does not write them): of scanlines, a block
// a line (as tinyexr takes them, whatever the compression says), or of tiles of one level;
// one to three channels of any type; up to three samples a pixel, now and then many.
std::vector<unsigned char> MakeDeep()
{
    static const char* const kDeepNames[6] = { "A", "B", "G", "R", "Z", "ZBack" };
    static const int kScanline[10] = { TINYEXR_COMPRESSIONTYPE_ZIPS, TINYEXR_COMPRESSIONTYPE_ZIPS, TINYEXR_COMPRESSIONTYPE_ZIPS, TINYEXR_COMPRESSIONTYPE_ZIPS, TINYEXR_COMPRESSIONTYPE_ZIPS, TINYEXR_COMPRESSIONTYPE_ZIP, TINYEXR_COMPRESSIONTYPE_NONE, TINYEXR_COMPRESSIONTYPE_RLE, TINYEXR_COMPRESSIONTYPE_PIZ, TINYEXR_COMPRESSIONTYPE_B44 };
    static const int kTiled[5] = { TINYEXR_COMPRESSIONTYPE_NONE, TINYEXR_COMPRESSIONTYPE_RLE, TINYEXR_COMPRESSIONTYPE_ZIPS, TINYEXR_COMPRESSIONTYPE_ZIP, TINYEXR_COMPRESSIONTYPE_PIZ };
    const bool tiled = Next() % 3 == 0;
    const int w = 1 + static_cast<int>(Next() % 20);
    const int h = 1 + static_cast<int>(Next() % 20);
    const int dx = Next() % 4 == 0 ? static_cast<int>(Next() % 5) : 0;
    const int dy = Next() % 4 == 0 ? static_cast<int>(Next() % 5) : 0;
    const int compression = tiled ? kTiled[Next() % 5] : kScanline[Next() % 10];
    const std::uint32_t many = Next() % 10 == 0 ? 40 : 4;
    const std::uint32_t num_channels = 1 + Next() % 3;
    const std::uint32_t first = Next() % (6 - num_channels + 1);
    std::vector<int> types;
    std::vector<unsigned char> f = { 0x76, 0x2f, 0x31, 0x01, 2, 8, 0, 0 };
    if (Next() % 30 == 0)
        f[5] = static_cast<unsigned char>(Next() % 32);
    {
        std::vector<unsigned char> chlist;
        for (std::uint32_t c = 0; c < num_channels; ++c)
        {
            const char* channel = kDeepNames[first + c];
            chlist.insert(chlist.end(), channel, channel + std::strlen(channel) + 1);
            types.push_back(static_cast<int>(Next() % 3));
            Put32(chlist, static_cast<std::uint32_t>(types.back()));
            Put32(chlist, 0);
            Put32(chlist, 1);
            Put32(chlist, 1);
        }
        chlist.push_back(0);
        PutAttribute(f, "channels", "chlist", chlist);
    }
    PutAttribute(f, "compression", "compression", { static_cast<unsigned char>(compression) });
    {
        std::vector<unsigned char> window;
        Put32(window, static_cast<std::uint32_t>(dx));
        Put32(window, static_cast<std::uint32_t>(dy));
        Put32(window, static_cast<std::uint32_t>(dx + w - 1));
        Put32(window, static_cast<std::uint32_t>(dy + h - 1));
        PutAttribute(f, "dataWindow", "box2i", window);
        PutAttribute(f, "displayWindow", "box2i", window);
    }
    PutAttribute(f, "lineOrder", "lineOrder", { 0 });
    int tx = 1, ty = 1;
    if (tiled)
    {
        PutAttribute(f, "type", "string", { 'd', 'e', 'e', 'p', 't', 'i', 'l', 'e' });
        tx = 1 + static_cast<int>(Next() % 8);
        ty = 1 + static_cast<int>(Next() % 8);
        std::vector<unsigned char> desc;
        Put32(desc, static_cast<std::uint32_t>(tx));
        Put32(desc, static_cast<std::uint32_t>(ty));
        desc.push_back(Next() % 20 == 0 ? 1 : 0);
        PutAttribute(f, "tiles", "tiledesc", desc);
    }
    f.push_back(0);

    // How many samples each pixel has, and their values.
    std::vector<std::uint32_t> counts(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (std::uint32_t& count : counts)
        count = Next() % many;
    auto sample = [&](std::vector<unsigned char>& to, int type) {
        const std::uint32_t value = Next();
        if (type == TINYEXR_PIXELTYPE_HALF)
        {
            to.push_back(static_cast<unsigned char>(value));
            to.push_back(static_cast<unsigned char>((value >> 8) & 0x7b));
        }
        else if (type == TINYEXR_PIXELTYPE_UINT)
            Put32(to, value % 1000);
        else
        {
            const float v = static_cast<float>(value % 10000) / 100.0f;
            std::uint32_t bits;
            std::memcpy(&bits, &v, 4);
            Put32(to, bits);
        }
    };
    auto block = [&](std::vector<unsigned char>& body, const std::vector<unsigned char>& table, const std::vector<unsigned char>& samples) {
        const std::vector<unsigned char> packed_table = Pack(compression, table);
        const std::vector<unsigned char> packed_samples = Pack(compression, samples);
        Put64(body, packed_table.size());
        Put64(body, packed_samples.size());
        Put64(body, samples.size());
        body.insert(body.end(), packed_table.begin(), packed_table.end());
        body.insert(body.end(), packed_samples.begin(), packed_samples.end());
    };

    std::vector<std::vector<unsigned char>> blocks;
    if (!tiled)
    {
        const int lines = compression == TINYEXR_COMPRESSIONTYPE_ZIP ? 16 : compression == TINYEXR_COMPRESSIONTYPE_B44 ? 32 : 1;
        const int num_blocks = (h + lines - 1) / lines;
        for (int b = 0; b < num_blocks; ++b)
        {
            std::vector<unsigned char> table, samples, body;
            std::uint32_t run = 0;
            for (int x = 0; x < w; ++x)
            {
                run += counts[static_cast<std::size_t>(b) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)];
                Put32(table, run);
            }
            for (std::uint32_t c = 0; c < num_channels; ++c)
                for (std::uint32_t s = 0; s < run; ++s)
                    sample(samples, types[c]);
            Put32(body, static_cast<std::uint32_t>(dy + b));
            block(body, table, samples);
            blocks.push_back(body);
        }
    }
    else
    {
        const int nx = (w + tx - 1) / tx, ny = (h + ty - 1) / ty;
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const int tw = std::min(tx, w - i * tx), th = std::min(ty, h - j * ty);
                std::vector<unsigned char> table, samples, body;
                std::uint32_t run = 0;
                for (int ry = 0; ry < th; ++ry)
                {
                    for (int rx = 0; rx < tw; ++rx)
                    {
                        run += counts[static_cast<std::size_t>(j * ty + ry) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i * tx + rx)];
                        Put32(table, run);
                    }
                }
                for (std::uint32_t c = 0; c < num_channels; ++c)
                    for (std::uint32_t s = 0; s < run; ++s)
                        sample(samples, types[c]);
                Put32(body, static_cast<std::uint32_t>(i));
                Put32(body, static_cast<std::uint32_t>(j));
                Put32(body, 0);
                Put32(body, 0);
                block(body, table, samples);
                blocks.push_back(body);
            }
        }
    }
    std::uint64_t at = f.size() + 8 * blocks.size();
    for (const std::vector<unsigned char>& body : blocks)
    {
        Put64(f, at);
        at += body.size();
    }
    for (const std::vector<unsigned char>& body : blocks)
        f.insert(f.end(), body.begin(), body.end());
    return f;
}

// The reading of a deep file, as examples/dump.rs does it, with `pad` outside it.
void ReadDeep(const char* path, std::size_t size, unsigned char pad)
{
    g_pad = 65536;
    g_pad_for = size;
    g_pad_byte = pad;
    DeepImage deep;
    std::memset(&deep, 0, sizeof deep);
    const char* err = NULL;
    const int ret = LoadDeepEXR(&deep, path, &err);
    g_pad = 0;
    std::unordered_map<void*, std::size_t> sizes;
    sizes.swap(g_sizes);
    AddNumber(static_cast<std::uint32_t>(ret));
    // (Its message names the file, which the port has no name for.)
    if (ret == TINYEXR_ERROR_INVALID_FILE)
    {
        AddString("File size is zero");
        FreeEXRErrorMessage(err);
    }
    else
        AddMessage2(err);
    if (ret != TINYEXR_SUCCESS)
        return;
    AddNumber(static_cast<std::uint32_t>(deep.width));
    AddNumber(static_cast<std::uint32_t>(deep.height));
    AddNumber(static_cast<std::uint32_t>(deep.num_channels));
    for (int c = 0; c < deep.num_channels; ++c)
        AddName(deep.channel_names[c]);
    if (deep.width <= 0 || deep.height <= 0)
        return;
    for (int y = 0; y < deep.height; ++y)
        for (int x = 0; x < deep.width; ++x)
            AddNumber(static_cast<std::uint32_t>(deep.offset_table[y][x]));
    for (int c = 0; c < deep.num_channels; ++c)
    {
        for (int y = 0; y < deep.height; ++y)
        {
            if (!deep.image[c][y])
                continue;
            AddNumber(static_cast<std::uint32_t>(y));
            // (A line of the samples there are, which a damaged file of scanlines can have
            // other than the count of: its allocation's. A file of tiles allocates a value for
            // a line of none.)
            std::size_t samples = sizes.count(deep.image[c][y]) ? sizes[deep.image[c][y]] / 4 : 0;
            if (samples == 1 && deep.offset_table[y][deep.width - 1] == 0)
                samples = 0;
            if (static_cast<std::size_t>(static_cast<std::uint32_t>(deep.offset_table[y][deep.width - 1])) != samples)
            {
                static const char kOther[] = "a line of other than its count of samples";
                Add(kOther, sizeof kOther - 1);
            }
            Add(deep.image[c][y], samples * 4);
        }
    }
}

// The `tinyexr_deep` dump: synthetic deep files, a third of them damaged, each read in a
// program of its own, with zeros and with ones outside it, and written to the file of files
// with what came of it.
int Deep(const char* name, std::uint32_t count, std::FILE* out)
{
    std::uint64_t file_total = 0;
    std::uint32_t past_end = 0, stopped = 0, left = 0;
    char path[64];
    std::snprintf(path, sizeof path, "/tmp/rustbcx-deep-%d.exr", static_cast<int>(getpid()));
    for (std::uint32_t i = 0; i < count; ++i)
    {
        std::vector<unsigned char> file = MakeDeep();
        // A third damaged: bits turned over, the end cut off, a byte randomised, a number made
        // up (one such as a damaged header has, or a small one).
        if (Next() % 3 == 0)
        {
            const std::uint32_t how = Next() % 5;
            if (how == 0)
            {
                for (std::uint32_t k = 1 + Next() % 3; k > 0; --k)
                {
                    const std::uint32_t bit = Next() % static_cast<std::uint32_t>(file.size() * 8);
                    file[bit >> 3] ^= static_cast<unsigned char>(1u << (bit & 7));
                }
            }
            else if (how == 1)
                file.resize(Next() % file.size());
            else if (how == 2)
                file[Next() % file.size()] = static_cast<unsigned char>(Next());
            else
            {
                const std::uint32_t at = Next() % static_cast<std::uint32_t>(file.size());
                SetNumber(file, at, how == 3 ? Special() : Next() % 64);
            }
        }
        file_total += file.size();
        AddNumber(static_cast<std::uint32_t>(file.size()));
        Add(file.data(), file.size());

        // (A data window of more pixels than there is time and memory for is left unread.)
        bool large = false;
        {
            const std::size_t at = FindAttribute(file, "dataWindow", "box2i");
            if (at && at + 16 <= file.size())
            {
                const std::int64_t w = static_cast<std::int64_t>(static_cast<std::int32_t>(GetNumber(file, at + 8))) - static_cast<std::int32_t>(GetNumber(file, at)) + 1;
                const std::int64_t h = static_cast<std::int64_t>(static_cast<std::int32_t>(GetNumber(file, at + 12))) - static_cast<std::int32_t>(GetNumber(file, at + 4)) + 1;
                large = w > (1 << 22) || h > (1 << 22) || w * h > (1 << 22);
            }
        }
        std::uint32_t flags = 4;
        if (large)
        {
            ++left;
            AddNumber(flags);
        }
        else
        {
            std::FILE* exr = std::fopen(path, "wb");
            if (!exr || (!file.empty() && std::fwrite(file.data(), 1, file.size(), exr) != file.size()) || std::fclose(exr) != 0)
            {
                std::fprintf(stderr, "cannot write %s\n", path);
                return 2;
            }
            const std::size_t size = file.size();
            const Outcome outcome = Isolated([&](unsigned char pad) { ReadDeep(path, size, pad); });
            flags = AddOutcome(outcome);
            past_end += flags == 1;
            stopped += flags == 2;
        }
        WriteNumber(out, flags);
        WriteNumber(out, static_cast<std::uint32_t>(file.size()));
        if (!file.empty())
            std::fwrite(file.data(), 1, file.size(), out);

#ifdef SHOW_EACH
        std::fprintf(stderr, "file %u %016llx\n", i, static_cast<unsigned long long>(g_hash));
#endif
        if ((i + 1) % 200 == 0 || i + 1 == count)
            std::printf("%s %u %016llx\n", name, i + 1, static_cast<unsigned long long>(g_hash));
    }
    std::remove(path);
    std::printf("%s bytes of files: %llu; read where it depends on what is outside the file: %u, where tinyexr stops: %u, left: %u\n", name, static_cast<unsigned long long>(file_total), past_end, stopped, left);
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
#ifdef SHOW_EACH
    if (argc >= 3 && std::strcmp(argv[1], "tinyexr_deep_one") == 0)
    {
        std::FILE* f = std::fopen(argv[2], "rb");
        std::fseek(f, 0, SEEK_END);
        const std::size_t size = static_cast<std::size_t>(std::ftell(f));
        std::fclose(f);
        const Outcome outcome = Isolated([&](unsigned char pad) { ReadDeep(argv[2], size, pad); });
        std::printf("%016llx %016llx stopped %u\n", static_cast<unsigned long long>(outcome.hash[0]), static_cast<unsigned long long>(outcome.hash[1]), outcome.stopped);
        return 0;
    }
#endif
    if (argc >= 5 && (std::strcmp(argv[1], "tinyexr_parts") == 0 || std::strcmp(argv[1], "tinyexr_deep") == 0))
    {
        g_state = std::strtoull(argv[2], NULL, 10);
        g_state2 = g_state ^ 0x9e3779b97f4a7c15ull;
        std::FILE* out = std::fopen(argv[4], "wb");
        if (!out)
        {
            std::fprintf(stderr, "cannot write %s\n", argv[4]);
            return 2;
        }
        const std::uint32_t count = static_cast<std::uint32_t>(std::strtoul(argv[3], NULL, 10));
        const int ret = std::strcmp(argv[1], "tinyexr_parts") == 0 ? Parts(argv[1], count, out) : Deep(argv[1], count, out);
        if (std::fclose(out) != 0)
        {
            std::fprintf(stderr, "cannot write %s\n", argv[4]);
            return 2;
        }
        return ret;
    }
    if (argc >= 4 && (std::strcmp(argv[1], "tinyexr_write") == 0 || std::strcmp(argv[1], "tinyexr_write_small") == 0 || std::strcmp(argv[1], "tinyexr_zfp_write") == 0))
    {
        g_state = std::strtoull(argv[2], NULL, 10);
        g_zfp = std::strcmp(argv[1], "tinyexr_zfp_write") == 0;
        return Write(argv[1], std::strcmp(argv[1], "tinyexr_write_small") == 0, static_cast<std::uint32_t>(std::strtoul(argv[3], NULL, 10)));
    }
    if (argc < 5 || (std::strcmp(argv[1], "tinyexr_read") != 0 && std::strcmp(argv[1], "tinyexr_read_small") != 0 && std::strcmp(argv[1], "tinyexr_zfp_read") != 0))
    {
        std::fprintf(stderr, "usage: dump tinyexr_read|tinyexr_read_small|tinyexr_zfp_read|tinyexr_parts|tinyexr_deep <seed> <count> <file>\n       dump tinyexr_write|tinyexr_write_small|tinyexr_zfp_write <seed> <count>\n");
        return 2;
    }
    const bool small = std::strcmp(argv[1], "tinyexr_read_small") == 0;
    g_zfp = std::strcmp(argv[1], "tinyexr_zfp_read") == 0;
    g_state = std::strtoull(argv[2], NULL, 10);
    const std::uint32_t count = static_cast<std::uint32_t>(std::strtoul(argv[3], NULL, 10));
    std::FILE* out = std::fopen(argv[4], "wb");
    if (!out)
    {
        std::fprintf(stderr, "cannot write %s\n", argv[4]);
        return 2;
    }
    Counts counts;
    std::uint64_t file_total = 0;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        Made made = MakeImage(small);
        std::string message;
        std::uint32_t refusal = 0;
        std::vector<unsigned char> file = Save(made, message, refusal);
        bool sound = !file.empty() && made.plain;
        bool damaged = false;
        // (What tinyexr does not save is read as a file of no bytes.)
        if (!file.empty())
        {
            // Now and then the order of the lines named the other way, or the window moved
            // across or down (which the blocks of a file of scanlines then do not agree with).
            const std::uint32_t change = Next() % 8;
            const std::uint32_t by = Next() % 101 - 50;
            if (change == 0)
            {
                const std::size_t at = FindAttribute(file, "lineOrder", "lineOrder");
                if (at)
                    file[at] = 1;
            }
            else if (change == 1 || change == 2)
            {
                const std::size_t at = FindAttribute(file, "dataWindow", "box2i");
                if (at)
                {
                    AddToNumber(file, at + (change == 1 ? 0 : 4), by);
                    AddToNumber(file, at + (change == 1 ? 8 : 12), by);
                }
                sound = sound && (change == 1 || made.tiled || by == 0);
            }
            // A third of them damaged: bits turned over, the end cut off, a byte randomised or
            // made zero, a number of the header randomised, the table of offsets changed, a
            // number at the start of a block randomised, or one of those a PIZ block's data
            // starts with.
            if (Next() % 3 == 0)
            {
                const std::uint32_t how = Next() % (g_zfp ? 10 : 9);
                sound = false;
                damaged = true;
                if (how == 0)
                {
                    for (std::uint32_t k = 1 + Next() % 3; k > 0; --k)
                    {
                        const std::uint32_t bit = Next() % static_cast<std::uint32_t>(file.size() * 8);
                        file[bit >> 3] ^= static_cast<unsigned char>(1u << (bit & 7));
                    }
                }
                else if (how == 1)
                {
                    // (Anywhere, or within its start, where the header and the table are.)
                    const std::uint32_t near = Next() % 2;
                    const std::uint32_t at = Next();
                    file.resize(at % (near ? std::min<std::size_t>(file.size(), 600) : file.size()));
                }
                else if (how == 2)
                {
                    const std::uint32_t at = Next() % static_cast<std::uint32_t>(file.size());
                    file[at] = static_cast<unsigned char>(Next());
                }
                else if (how == 3)
                {
                    const std::uint32_t bit = Next() % static_cast<std::uint32_t>(std::min<std::size_t>(file.size(), 400) * 8);
                    file[bit >> 3] ^= static_cast<unsigned char>(1u << (bit & 7));
                }
                else if (how == 4)
                    file[Next() % static_cast<std::uint32_t>(std::min<std::size_t>(file.size(), 400))] = 0;
                else if (how == 5)
                {
                    const std::uint32_t which = Next() % 8;
                    const std::uint32_t pick = Next();
                    const std::uint32_t value = Special();
                    if (which < 3)
                    {
                        const std::size_t at = FindAttribute(file, which < 2 ? "dataWindow" : "displayWindow", "box2i");
                        if (at)
                            SetNumber(file, at + 4 * (pick % 4), value);
                    }
                    else if (which == 3)
                    {
                        const std::size_t at = FindAttribute(file, "tiles", "tiledesc");
                        if (at && pick % 3 < 2)
                            SetNumber(file, at + 4 * (pick % 3), value);
                        else if (at)
                            file[at + 8] = static_cast<unsigned char>(pick >> 8);
                    }
                    else if (which == 4)
                    {
                        const std::size_t at = FindAttribute(file, "compression", "compression");
                        if (at)
                            file[at] = static_cast<unsigned char>(pick % 13 == 12 ? 128 : pick % 13);
                    }
                    else if (which == 5)
                    {
                        // The first channel: its name, then its type, a byte and three more,
                        // and its two samplings.
                        std::size_t at = FindAttribute(file, "channels", "chlist");
                        if (at)
                        {
                            while (file[at])
                                ++at;
                            SetNumber(file, at + 1 + (pick % 3 == 0 ? 0 : pick % 3 == 1 ? 8 : 12), (pick >> 8) % 4);
                        }
                    }
                    else if (which == 6)
                    {
                        const std::size_t at = FindAttribute(file, "chunkCount", "int");
                        if (at)
                            SetNumber(file, at, value);
                    }
                    else
                    {
                        const std::size_t at = FindAttribute(file, "lineOrder", "lineOrder");
                        if (at)
                            file[at] = static_cast<unsigned char>(pick % 4);
                    }
                }
                else if (how == 9)
                {
                    // (ZFP) one of its settings randomised.
                    const std::uint32_t which = Next() % 4;
                    if (which == 0)
                    {
                        const std::size_t at = FindAttribute(file, "zfpCompressionType", "uchar");
                        const std::uint32_t type = Next() % 5;
                        if (at)
                            file[at] = static_cast<unsigned char>(type);
                    }
                    else
                    {
                        const char* name = which == 1 ? "zfpCompressionRate" : which == 2 ? "zfpCompressionTolerance" : "zfpCompressionPrecision";
                        const std::size_t at = FindAttribute(file, name, which == 3 ? "int" : "double");
                        const double value = ZfpDouble(which == 2, made.channels.size() > 1);
                        const std::uint32_t precision = Next();
                        if (at && which < 3 && at + 8 <= file.size())
                            std::memcpy(&file[at], &value, 8);
                        else if (at && at + 4 <= file.size())
                            std::memcpy(&file[at], &precision, 4);
                    }
                }
                else
                {
                    // The table of offsets comes after the header: one of its first six
                    // made zero, put near the end, randomised, or changed for the one after
                    // it; or a number at the start of the block it leads to randomised.
                    const std::size_t table = EndOfHeader(file);
                    const std::uint32_t entry = Next() % 6;
                    const std::uint32_t pick = Next();
                    const std::uint32_t value = Special();
                    const std::size_t at = table + 8 * entry;
                    if (table && at + 16 <= file.size())
                    {
                        if (how == 6)
                        {
                            switch (pick % 4)
                            {
                                case 0:
                                    SetNumber(file, at, 0);
                                    SetNumber(file, at + 4, 0);
                                    break;
                                case 1:
                                    SetNumber(file, at, static_cast<std::uint32_t>(file.size()) - (pick >> 8) % 16);
                                    break;
                                case 2:
                                    SetNumber(file, at, (pick >> 8) % static_cast<std::uint32_t>(file.size()));
                                    break;
                                default:
                                    for (std::size_t k = 0; k < 8; ++k)
                                        std::swap(file[at + k], file[at + 8 + k]);
                                    break;
                            }
                        }
                        else
                        {
                            const std::size_t block = GetNumber(file, at);
                            if (GetNumber(file, at + 4) == 0 && block + 20 <= file.size())
                            {
                                if (how == 7 || made.compression != TINYEXR_COMPRESSIONTYPE_PIZ)
                                    SetNumber(file, block + 4 * (pick % 5), value);
                                else
                                {
                                    // A PIZ block: after the numbers it starts with, the
                                    // least and the most of its values, a bitmap from the one
                                    // to the other, a length, and the five numbers the
                                    // Huffman coder starts with: one of the six randomised.
                                    std::size_t data = block + (made.tiled ? 20 : 8);
                                    if (data + 4 <= file.size())
                                    {
                                        const std::uint32_t least = static_cast<std::uint32_t>(file[data]) | static_cast<std::uint32_t>(file[data + 1]) << 8;
                                        const std::uint32_t most = static_cast<std::uint32_t>(file[data + 2]) | static_cast<std::uint32_t>(file[data + 3]) << 8;
                                        data += 4 + (least <= most ? most - least + 1 : 0);
                                        SetNumber(file, data + 4 * (pick % 6), (pick >> 8) % 4 == 0 ? value : (pick >> 10) % 70000);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // What to read it as: halves as they are, or as floats; and whether it was damaged.
        const std::uint32_t wish = (Next() % 2) | (damaged ? 2u : 0u);
        file_total += file.size();

        AddNumber(static_cast<std::uint32_t>(file.size()));
        Add(file.data(), file.size());
        // (Read from memory of just the file's size, so that a sanitizer sees a read past
        // its end. A file of no bytes is still a file: tinyexr has other answers for no
        // pointer.)
        unsigned char* exact = static_cast<unsigned char*>(std::malloc(file.empty() ? 1 : file.size()));
        if (!exact)
            return 2;
        if (!file.empty())
            std::memcpy(exact, file.data(), file.size());
        std::uint32_t stops = 0;
        const bool read = Read(exact, file.size(), wish, counts, stops);
        std::free(exact);

        WriteNumber(out, wish | stops);
        WriteNumber(out, static_cast<std::uint32_t>(file.size()));
        if (!file.empty())
            std::fwrite(file.data(), 1, file.size(), out);
#ifdef SHOW_EACH
        std::fprintf(stderr, "file %u %016llx\n", i, static_cast<unsigned long long>(g_hash));
#endif
        // A sound file has to be read.
        if (sound && !read)
        {
            std::fprintf(stderr, "%s: file %u is sound and tinyexr does not read it\n", argv[1], i);
            return 2;
        }

        if ((i + 1) % 200 == 0 || i + 1 == count)
            std::printf("%s %u %016llx\n", argv[1], i + 1, static_cast<unsigned long long>(g_hash));
    }
    if (std::fclose(out) != 0)
    {
        std::fprintf(stderr, "cannot write %s\n", argv[4]);
        return 2;
    }
    std::printf("%s bytes of files: %llu; headers read: %u, images read: %u, refused: %u, left: %u, where tinyexr stops: %u; as red, green, blue and alpha: %u\n", argv[1], static_cast<unsigned long long>(file_total), counts.headers, counts.images, counts.refused, counts.left, counts.stopped, counts.as_rgba);
    return 0;
}
