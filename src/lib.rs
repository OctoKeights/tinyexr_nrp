//! `.exr` files (OpenEXR), read and written: a native Rust port of Syoyo Fujita's
//! [tinyexr](https://github.com/syoyo/tinyexr), v3.4.0 (`tinyexr.h`), as built without
//! threads and with miniz for its ZIP compression ([`miniz_nrp`]).
//!
//! A file of one part, of scanlines or of tiles (with mip or rip levels), whose pixels are
//! uncompressed or compressed with RLE, ZIPS, ZIP, PIZ, PXR24, B44 or B44A, or with DWAA or
//! DWAB, which tinyexr does not have. Its channels are halves, floats or 32-bit integers.
//!
//! - [`load_exr_from_memory`] gives red, green, blue and alpha as floats, and
//!   [`save_exr_to_memory`] writes floats of one, three or four to a pixel: what most
//!   callers want.
//! - [`parse_exr_version_from_memory`], [`parse_exr_header_from_memory`] and
//!   [`load_exr_image_from_memory`] give the header and then every channel as it is stored
//!   (or the halves as floats), or every tile of every level;
//!   [`save_exr_image_to_memory`] writes an image the same way, with [`tile_levels`] for
//!   the levels a tiled one has.
//!
//! The port writes the files the original writes, byte for byte, gives the values it gives,
//! and refuses the files it refuses with the same code and message ([`ExrError`]), with
//! these exceptions:
//!
//! - **Blocks that do not come out smaller.** The format has a block stored as its pixels
//!   when compressing it does not make it smaller, and a reader take a block as long as its
//!   pixels for them. tinyexr does both for RLE, ZIP and PIZ and neither for PXR24, B44 and
//!   B44A, so that other readers misread some of its files (every B44 file without a channel
//!   of halves, and blocks of a few pixels), it misreads some of theirs, and now and then one
//!   of its own PXR24 blocks. Here the format is followed; the functions that end in
//!   `_as_tinyexr` do exactly as tinyexr does.
//! - **Damaged files that stop the original**, or have it read past the end of the file, are
//!   refused: a tile size of 0, a count of chunks larger than the file can hold a table for,
//!   a compressed tile whose place is past the image's edge, and a block that is not
//!   compressed and is shorter than its lines.
//! - **What the writer is handed** is looked at where the original trusts it: a channel with
//!   too few values, or a type that is not the values' type, is refused.
//! - A small PIZ block of many different values, which the original writes past its buffer
//!   for, is written as the original would write it with room enough.
//! - **DWAA and DWAB**, which tinyexr refuses ("Unknown compression type."), are read and
//!   written as OpenEXR 3.5 reads and writes them on an x86-64 machine with AVX
//!   ([`openexr_nrp::dwa`]),
//!   at the level of loss of the header's `dwaCompressionLevel`
//!   ([`dwa_compression_level_attribute`]; 45 without it). The functions that end in
//!   `_as_tinyexr`, [`parse_exr_header_from_memory_as_tinyexr`] and
//!   [`parse_exr_multipart_header_from_memory_as_tinyexr`] among them, refuse them as tinyexr
//!   does.
//!
//! Files of several parts are read and written
//! ([`parse_exr_multipart_header_from_memory`], [`load_exr_multipart_image_from_memory`],
//! [`save_exr_multipart_image_to_memory`]), and deep images read
//! ([`load_deep_exr_from_memory`]; tinyexr does not write them). Their readers in the original
//! read past the end of a damaged file without a look. The port makes those reads with zeros
//! and then with ones past the end: where the answer is the same, it is the original's answer
//! whatever its memory holds, and it is given; where it is not, or the original would stop,
//! the file is refused with [`PAST_THE_END`].
//!
//! The ZFP compression tinyexr adds to the format is read and written, as tinyexr does with
//! `TINYEXR_USE_ZFP`, with the port of what it uses of the zfp library v1.0.1 ([`zfp_nrp`]): for
//! floats only, with the settings in attributes of the header ([`ZfpSettings`] makes them). What
//! tinyexr cannot write with it is refused with a message of the port's own: a block or a tile that
//! is not of whole blocks of four by four ([`ZFP_NOT_MULTIPLE_OF_FOUR`]; tinyexr writes a length it
//! never set), and settings that ask for more than a gigabyte for a block ([`ZFP_TOO_LARGE`]).

#![forbid(unsafe_code)]
// The code follows the original's line for line, and these are its shapes.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_range_contains
)]
// (The arithmetic is written as the original has it.)
#![allow(clippy::if_same_then_else, clippy::manual_div_ceil)]

mod b44;
mod chunk;
mod decode;
mod deep;
mod encode;
mod parts;
mod piz;
mod pxr24;
mod zfp;
mod zip_rle;

pub use chunk::{TileLevel, tile_levels};
pub use deep::{DeepImage, load_deep_exr_from_memory};
pub use encode::{
    DWA_INVALID_LEVEL, ZFP_NOT_MULTIPLE_OF_FOUR, ZFP_TOO_LARGE, dwa_compression_level_attribute,
    save_exr_image_to_memory, save_exr_image_to_memory_as_tinyexr,
    save_exr_multipart_image_to_memory, save_exr_multipart_image_to_memory_as_tinyexr,
    save_exr_to_memory,
};
pub use parts::{
    PAST_THE_END, load_exr_multipart_image_from_memory,
    load_exr_multipart_image_from_memory_as_tinyexr, parse_exr_multipart_header_from_memory,
    parse_exr_multipart_header_from_memory_as_tinyexr,
};
pub use zfp::{
    ZFP_COMPRESSIONTYPE_ACCURACY, ZFP_COMPRESSIONTYPE_PRECISION, ZFP_COMPRESSIONTYPE_RATE,
    ZfpSettings,
};

/// The original's error codes (`TINYEXR_ERROR_*`).
pub const ERROR_INVALID_MAGIC_NUMBER: i32 = -1;
pub const ERROR_INVALID_EXR_VERSION: i32 = -2;
pub const ERROR_INVALID_ARGUMENT: i32 = -3;
pub const ERROR_INVALID_DATA: i32 = -4;
pub const ERROR_INVALID_FILE: i32 = -5;
pub const ERROR_INVALID_PARAMETER: i32 = -6;
pub const ERROR_UNSUPPORTED_FORMAT: i32 = -8;
pub const ERROR_INVALID_HEADER: i32 = -9;
pub const ERROR_UNSUPPORTED_FEATURE: i32 = -10;
pub const ERROR_SERIALIZATION_FAILED: i32 = -12;
pub const ERROR_LAYER_NOT_FOUND: i32 = -13;
pub const ERROR_DATA_TOO_LARGE: i32 = -14;

/// The types of a channel's values (`TINYEXR_PIXELTYPE_*`).
pub const PIXELTYPE_UINT: i32 = 0;
pub const PIXELTYPE_HALF: i32 = 1;
pub const PIXELTYPE_FLOAT: i32 = 2;

const MAX_HEADER_ATTRIBUTES: usize = 1024;
const MAX_CUSTOM_ATTRIBUTES: usize = 128;

/// How the pixels of a file are compressed (`TINYEXR_COMPRESSIONTYPE_*`).
pub const COMPRESSIONTYPE_NONE: i32 = 0;
pub const COMPRESSIONTYPE_RLE: i32 = 1;
pub const COMPRESSIONTYPE_ZIPS: i32 = 2;
pub const COMPRESSIONTYPE_ZIP: i32 = 3;
pub const COMPRESSIONTYPE_PIZ: i32 = 4;
pub const COMPRESSIONTYPE_PXR24: i32 = 5;
pub const COMPRESSIONTYPE_B44: i32 = 6;
pub const COMPRESSIONTYPE_B44A: i32 = 7;
/// DreamWorks Animation's lossy compressions, of 32 lines and of 256 lines to a block (which
/// tinyexr does not have: the functions that end in `_as_tinyexr` refuse them as it does).
/// The `dwaCompressionLevel` attribute of a header, which
/// [`dwa_compression_level_attribute`] makes, says how lossy (45 if a header has none).
pub const COMPRESSIONTYPE_DWAA: i32 = 8;
pub const COMPRESSIONTYPE_DWAB: i32 = 9;
/// A tinyexr extension, of floats only, with the zfp library's coder; its settings are in
/// attributes of the header, which [`ZfpSettings`] makes.
pub const COMPRESSIONTYPE_ZFP: i32 = 128;

/// The levels of a tiled file (`TINYEXR_TILE_*`).
pub const TILE_ONE_LEVEL: i32 = 0;
pub const TILE_MIPMAP_LEVELS: i32 = 1;
pub const TILE_RIPMAP_LEVELS: i32 = 2;
pub const TILE_ROUND_DOWN: i32 = 0;
pub const TILE_ROUND_UP: i32 = 1;

/// Why a file was refused: the original's error code and the message it gives with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExrError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for ExrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() {
            write!(
                f,
                "the .exr file could not be read or written (tinyexr's error {})",
                self.code
            )
        } else {
            f.write_str(self.message.trim_end())
        }
    }
}

impl std::error::Error for ExrError {}

/// `EXRVersion`: the eight bytes a file starts with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExrVersion {
    /// This will be 2.
    pub version: i32,
    /// Tile format image; not zero only for a single-part "normal" image (not deep, not
    /// multipart).
    pub tiled: bool,
    /// Long name attribute.
    pub long_name: bool,
    /// Deep image (EXR 2.0); zero for an OpenEXR 1.x image.
    pub non_image: bool,
    /// Multi-part (EXR 2.0); zero for an OpenEXR 1.x image.
    pub multipart: bool,
}

/// `EXRAttribute`: an attribute of the header that the reader has no field for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Attribute {
    pub name: String,
    /// The name of its type, such as `"string"` or `"box2i"`.
    pub type_name: String,
    /// Its value as the file has it.
    pub value: Vec<u8>,
}

/// `EXRChannelInfo`: a channel of a file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChannelInfo {
    pub name: String,
    /// One of the `PIXELTYPE_*`.
    pub pixel_type: i32,
    pub x_sampling: i32,
    pub y_sampling: i32,
    pub p_linear: u8,
}

/// `EXRBox2i`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Box2i {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

/// The original's `HeaderInfo`: a header as it is parsed, before it becomes an `ExrHeader`.
#[derive(Clone, Debug, Default)]
pub(crate) struct HeaderInfo {
    channels: Vec<ChannelInfo>,
    attributes: Vec<Attribute>,

    data_window: Box2i,
    line_order: i32,
    display_window: Box2i,
    screen_window_center: [f32; 2],
    screen_window_width: f32,
    pixel_aspect_ratio: f32,

    chunk_count: i32,

    // Tiled format
    tiled: i32, // Non-zero if the part is tiled.
    tile_size_x: i32,
    tile_size_y: i32,
    tile_level_mode: i32,
    tile_rounding_mode: i32,

    header_len: u32,

    compression_type: i32,

    // required for multi-part or non-image files
    name: String,
    // required for multi-part or non-image files
    type_name: String,
}

/// The original's `FP16` and `half_to_float`: a half as a float, by shifting its bits into
/// place and, for the smallest values, subtracting where a float's hidden bit would be.
pub fn half_to_float(h: u16) -> f32 {
    let magic = f32::from_bits(113 << 23);
    const SHIFTED_EXP: u32 = 0x7c00 << 13; // exponent mask after shift
    let h = h as u32;

    let mut o = (h & 0x7fff) << 13; // exponent/mantissa bits
    let exp_ = SHIFTED_EXP & o; // just the exponent
    o = o.wrapping_add((127 - 15) << 23); // exponent adjust

    // handle exponent special cases
    if exp_ == SHIFTED_EXP {
        // Inf/NaN?
        o = o.wrapping_add((128 - 16) << 23); // extra exp adjust
    } else if exp_ == 0 {
        // Zero/Denormal?
        o = o.wrapping_add(1 << 23); // extra exp adjust
        o = (f32::from_bits(o) - magic).to_bits(); // renormalize
    }

    o |= (h & 0x8000) << 16; // sign bit
    f32::from_bits(o)
}

/// The original's `float_to_half_full`: a float as a half, rounded to the nearest with a
/// half going up.
pub fn float_to_half_full(f: f32) -> u16 {
    let bits = f.to_bits();
    let f_mantissa = bits & 0x7fffff;
    let f_exponent = (bits >> 23) & 0xff;
    let f_sign = bits >> 31;
    // The half as its fields: the original's `o.s`, where `o.u++` carries from the mantissa
    // into the exponent.
    let mut o: u16 = 0;
    let set = |o: &mut u16, exponent: u32, mantissa: u32| {
        *o = (*o & 0x8000) | (((exponent & 0x1f) as u16) << 10) | (mantissa & 0x3ff) as u16
    };

    // Based on ISPC reference code (with minor modifications).
    if f_exponent == 0 {
        // Signed zero/denormal (which will underflow).
        set(&mut o, 0, 0);
    } else if f_exponent == 255 {
        // Inf or NaN (all exponent bits set).
        set(&mut o, 31, if f_mantissa != 0 { 0x200 } else { 0 }); // NaN->qNaN and Inf->Inf
    } else {
        // Normalized number.
        // Exponent unbias the single, then bias the halfp.
        let newexp = f_exponent as i32 - 127 + 15;
        if newexp >= 31 {
            // Overflow, return signed infinity.
            set(&mut o, 31, 0);
        } else if newexp <= 0 {
            // Underflow.
            if (14 - newexp) <= 24 {
                // Mantissa might be non-zero.
                let mant = f_mantissa | 0x800000; // Hidden 1 bit
                set(&mut o, 0, mant >> (14 - newexp));
                if (mant >> (13 - newexp)) & 1 != 0 {
                    // Check for rounding.
                    o = o.wrapping_add(1); // Round, might overflow into exp bit, but this is OK
                }
            }
        } else {
            set(&mut o, newexp as u32, f_mantissa >> 13);
            if f_mantissa & 0x1000 != 0 {
                // Check for rounding.
                o = o.wrapping_add(1); // Round, might overflow to inf, this is OK
            }
        }
    }

    (o & 0x7fff) | ((f_sign as u16) << 15)
}

/// `ReadString`: a string that ends in a zero within the `len` bytes at `ptr` of `data`.
/// The answer is the string and where the byte after its zero is, or `None` when there is
/// no zero.
fn read_string(data: &[u8], ptr: usize, len: usize) -> Option<(String, usize)> {
    let mut q = ptr;
    while q - ptr < len && data[q] != 0 {
        q += 1;
    }

    if q - ptr >= len {
        return None;
    }

    Some((bytes_to_string(&data[ptr..q]), q + 1)) // skip '\0'
}

/// The bytes of a name as a string. (The original keeps them as bytes; names that are not
/// UTF-8 lose the bytes that are not.)
fn bytes_to_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `ReadAttribute`: the attribute at the start of `marker`: its name, the name of its type,
/// its value, and how many bytes all of it took.
pub(crate) fn read_attribute(marker: &[u8]) -> Option<(String, String, Vec<u8>, usize)> {
    let mut size = marker.len();
    let mut at = 0usize;
    let strnlen = |from: usize, size: usize| {
        marker[from..from + size]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(size)
    };

    let name_len = strnlen(at, size);
    if name_len == size {
        // String does not have a terminating character.
        return None;
    }
    let name = bytes_to_string(&marker[at..at + name_len]);

    at += name_len + 1;
    size -= name_len + 1;

    let type_len = strnlen(at, size);
    if type_len == size {
        return None;
    }
    let type_name = bytes_to_string(&marker[at..at + type_len]);

    at += type_len + 1;
    size -= type_len + 1;

    if size < 4 {
        return None;
    }

    let data_len =
        u32::from_le_bytes([marker[at], marker[at + 1], marker[at + 2], marker[at + 3]]) as usize;

    if data_len == 0 {
        if type_name == "string" {
            // Accept empty string attribute.
            return Some((name, type_name, vec![0], name_len + 1 + type_len + 1 + 4));
        } else {
            return None;
        }
    }

    at += 4;
    size -= 4;

    if size < data_len {
        return None;
    }

    let data = marker[at..at + data_len].to_vec();

    Some((
        name,
        type_name,
        data,
        name_len + 1 + type_len + 1 + 4 + data_len,
    ))
}

/// `ReadChannelInfo`: the channels of a `channels` attribute, added to `channels`.
pub(crate) fn read_channel_info(channels: &mut Vec<ChannelInfo>, data: &[u8]) -> bool {
    let mut p = 0usize;

    loop {
        if data[p] == 0 {
            break;
        }
        let data_len = data.len() - p;

        let Some((name, after)) = read_string(data, p, data_len) else {
            // Buffer overrun. Issue #51.
            return false;
        };
        p = after;

        let data_end = p + 16;
        if data_end >= data.len() {
            return false;
        }

        let int_at =
            |at: usize| i32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        let pixel_type = int_at(p);
        p += 4;
        let p_linear = data[p];
        p += 1 + 3; // reserved: uchar[3]
        let x_sampling = int_at(p);
        p += 4;
        let y_sampling = int_at(p);
        p += 4;

        channels.push(ChannelInfo {
            name,
            pixel_type,
            x_sampling,
            y_sampling,
            p_linear,
        });
    }

    true
}

/// `ParseEXRHeader`: the attributes of a header at the start of `buf`. In a file of many
/// parts, an empty header (a lone zero byte) is the end of the list of headers: then
/// `empty_header` is set and `info` is left as it is.
pub(crate) fn parse_exr_header(
    info: &mut HeaderInfo,
    empty_header: &mut bool,
    version: &ExrVersion,
    err: &mut String,
    buf: &[u8],
    as_tinyexr: bool,
) -> Result<(), i32> {
    let mut size = buf.len();
    let mut marker = 0usize;

    *empty_header = false;

    if version.multipart && size > 0 && buf[marker] == 0 {
        // End of header list.
        *empty_header = true;
        return Ok(());
    }

    // According to the spec, the header of every OpenEXR file must contain at least the
    // following attributes:
    //
    // channels chlist
    // compression compression
    // dataWindow box2i
    // displayWindow box2i
    // lineOrder lineOrder
    // pixelAspectRatio float
    // screenWindowCenter v2f
    // screenWindowWidth float
    let mut has_channels = false;
    let mut has_compression = false;
    let mut has_data_window = false;
    let mut has_display_window = false;
    let mut has_line_order = false;
    let mut has_pixel_aspect_ratio = false;
    let mut has_screen_window_center = false;
    let mut has_screen_window_width = false;
    let mut has_name = false;
    let mut has_type = false;

    info.name.clear();
    info.type_name.clear();

    info.data_window = Box2i::default();
    info.line_order = 0; // @fixme
    info.display_window = Box2i::default();
    info.screen_window_center = [0.0, 0.0];
    info.screen_window_width = -1.0;
    info.pixel_aspect_ratio = -1.0;

    info.tiled = 0;
    info.tile_size_x = -1;
    info.tile_size_y = -1;
    info.tile_level_mode = -1;
    info.tile_rounding_mode = -1;

    info.attributes.clear();

    // Read attributes.
    let orig_size = size;
    for _nattr in 0..MAX_HEADER_ATTRIBUTES {
        if 0 == size {
            err.push_str("Insufficient data size for attributes.\n");
            return Err(ERROR_INVALID_DATA);
        } else if buf[marker] == 0 {
            size -= 1;
            break;
        }

        let Some((attr_name, attr_type, mut data, marker_size)) =
            read_attribute(&buf[marker..marker + size])
        else {
            err.push_str("Failed to read attribute.\n");
            return Err(ERROR_INVALID_DATA);
        };
        marker += marker_size;
        size -= marker_size;

        let int_at = |data: &[u8], at: usize| {
            i32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
        };
        let float_at = |data: &[u8], at: usize| f32::from_bits(int_at(data, at) as u32);
        let box_at = |data: &[u8]| Box2i {
            min_x: int_at(data, 0),
            min_y: int_at(data, 4),
            max_x: int_at(data, 8),
            max_y: int_at(data, 12),
        };

        // For a multipart file, the version field 9th bit is 0.
        if (version.tiled || version.multipart || version.non_image) && attr_name == "tiles" {
            if data.len() != 9 {
                err.push_str("(ParseEXRHeader) Invalid attribute data size. Attribute data size must be 9.\n");
                return Err(ERROR_INVALID_DATA);
            }

            let x_size = int_at(&data, 0) as u32;
            let y_size = int_at(&data, 4) as u32;
            let tile_mode = data[8];

            if x_size > i32::MAX as u32 || y_size > i32::MAX as u32 {
                *err = "Tile sizes were invalid.".to_string();
                return Err(ERROR_UNSUPPORTED_FORMAT);
            }
            info.tile_size_x = x_size as i32;
            info.tile_size_y = y_size as i32;

            // mode = levelMode + roundingMode * 16
            info.tile_level_mode = (tile_mode & 0x3) as i32;
            info.tile_rounding_mode = ((tile_mode >> 4) & 0x1) as i32;
            info.tiled = 1;
        } else if attr_name == "compression" {
            let mut ok = false;
            let compression = data[0] as i32;
            if compression < COMPRESSIONTYPE_PIZ {
                ok = true;
            }

            if compression == COMPRESSIONTYPE_PIZ {
                ok = true;
            }

            if compression == COMPRESSIONTYPE_ZFP {
                ok = true;
            }

            if compression == COMPRESSIONTYPE_PXR24
                || compression == COMPRESSIONTYPE_B44
                || compression == COMPRESSIONTYPE_B44A
            {
                ok = true;
            }

            // (Not in the original.)
            if !as_tinyexr
                && (compression == COMPRESSIONTYPE_DWAA || compression == COMPRESSIONTYPE_DWAB)
            {
                ok = true;
            }

            if !ok {
                *err = "Unknown compression type.".to_string();
                return Err(ERROR_UNSUPPORTED_FORMAT);
            }

            info.compression_type = compression;
            has_compression = true;
        } else if attr_name == "channels" {
            // name: zero-terminated string, from 1 to 255 bytes long
            // pixel type: int, possible values are: UINT = 0 HALF = 1 FLOAT = 2
            // pLinear: unsigned char, possible values are 0 and 1
            // reserved: three chars, should be zero
            // xSampling: int
            // ySampling: int
            if !read_channel_info(&mut info.channels, &data) {
                err.push_str("Failed to parse channel info.\n");
                return Err(ERROR_INVALID_DATA);
            }

            if info.channels.is_empty() {
                err.push_str("# of channels is zero.\n");
                return Err(ERROR_INVALID_DATA);
            }

            has_channels = true;
        } else if attr_name == "dataWindow" {
            if data.len() >= 16 {
                info.data_window = box_at(&data);
                has_data_window = true;
            }
        } else if attr_name == "displayWindow" {
            if data.len() >= 16 {
                info.display_window = box_at(&data);
                has_display_window = true;
            }
        } else if attr_name == "lineOrder" {
            if !data.is_empty() {
                info.line_order = data[0] as i32;
                has_line_order = true;
            }
        } else if attr_name == "pixelAspectRatio" {
            if data.len() >= 4 {
                info.pixel_aspect_ratio = float_at(&data, 0);
                has_pixel_aspect_ratio = true;
            }
        } else if attr_name == "screenWindowCenter" {
            if data.len() >= 8 {
                info.screen_window_center = [float_at(&data, 0), float_at(&data, 4)];
                has_screen_window_center = true;
            }
        } else if attr_name == "screenWindowWidth" {
            if data.len() >= 4 {
                info.screen_window_width = float_at(&data, 0);
                has_screen_window_width = true;
            }
        } else if attr_name == "chunkCount" {
            if data.len() >= 4 {
                info.chunk_count = int_at(&data, 0);
            }
        } else if attr_name == "name" {
            if !data.is_empty() && data[0] != 0 {
                data.push(0);
                let len = data.iter().position(|&b| b == 0).unwrap_or(0);
                info.name = bytes_to_string(&data[..len]);
                has_name = true;
            }
        } else if attr_name == "type" {
            if !data.is_empty() && data[0] != 0 {
                data.push(0);
                let len = data.iter().position(|&b| b == 0).unwrap_or(0);
                info.type_name = bytes_to_string(&data[..len]);
                has_type = true;
            }
        } else {
            // Custom attribute (up to TINYEXR_MAX_CUSTOM_ATTRIBUTES).
            if info.attributes.len() < MAX_CUSTOM_ATTRIBUTES {
                // (The original keeps 255 bytes of each name.)
                let keep = |s: &str| bytes_to_string(&s.as_bytes()[..s.len().min(255)]);
                info.attributes.push(Attribute {
                    name: keep(&attr_name),
                    type_name: keep(&attr_type),
                    value: data,
                });
            }
        }
    }

    // Check if required attributes exist.
    {
        let mut ss_err = String::new();

        if !has_compression {
            ss_err.push_str("\"compression\" attribute not found in the header.\n");
        }

        if !has_channels {
            ss_err.push_str("\"channels\" attribute not found in the header.\n");
        }

        if !has_line_order {
            ss_err.push_str("\"lineOrder\" attribute not found in the header.\n");
        }

        if !has_display_window {
            ss_err.push_str("\"displayWindow\" attribute not found in the header.\n");
        }

        if !has_data_window {
            ss_err.push_str("\"dataWindow\" attribute not found in the header or invalid.\n");
        }

        if !has_pixel_aspect_ratio {
            ss_err.push_str("\"pixelAspectRatio\" attribute not found in the header.\n");
        }

        if !has_screen_window_width {
            ss_err.push_str("\"screenWindowWidth\" attribute not found in the header.\n");
        }

        if !has_screen_window_center {
            ss_err.push_str("\"screenWindowCenter\" attribute not found in the header.\n");
        }

        if version.multipart || version.non_image {
            if !has_name {
                ss_err.push_str("\"name\" attribute not found in the header.\n");
            }
            if !has_type {
                ss_err.push_str("\"type\" attribute not found in the header.\n");
            }
        }

        if !ss_err.is_empty() {
            err.push_str(&ss_err);
            return Err(ERROR_INVALID_HEADER);
        }
    }

    info.header_len = (orig_size - size) as u32;

    Ok(())
}

pub use decode::ChannelData;

/// `EXRHeader`: what the header of a file says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExrHeader {
    pub pixel_aspect_ratio: f32,
    pub line_order: i32,
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub screen_window_center: [f32; 2],
    pub screen_window_width: f32,

    pub chunk_count: i32,

    // Properties for tiled format (`tiledesc`).
    pub tiled: i32,
    pub tile_size_x: i32,
    pub tile_size_y: i32,
    /// One of the `TILE_*_LEVELS`.
    pub tile_level_mode: i32,
    /// `TILE_ROUND_DOWN` or `TILE_ROUND_UP`.
    pub tile_rounding_mode: i32,

    pub long_name: i32,
    // For a single-part file, agree with the version field bit 11; for a multi-part file,
    // agree with each OpenEXR Data Type name ("deepscanline" or "deeptile").
    pub non_image: i32,
    pub multipart: i32,
    pub header_len: u32,

    /// The attributes the reader has no field for (up to 128 of them).
    pub custom_attributes: Vec<Attribute>,

    pub channels: Vec<ChannelInfo>,

    /// The type of each channel's values as they were loaded: its type in the file until
    /// the image is read, the type that was asked for afterwards.
    pub pixel_types: Vec<i32>,

    /// One of the `COMPRESSIONTYPE_*`.
    pub compression_type: i32,
    /// The type to read each channel's values as: its type in the file, which can be
    /// changed to `PIXELTYPE_FLOAT` for a channel of halves before the image is read.
    pub requested_pixel_types: Vec<i32>,

    /// The name attribute, which a multipart file's parts need.
    pub name: String,
}

/// `EXRTile`: a tile of a tiled image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExrTile {
    pub offset_x: i32,
    pub offset_y: i32,
    pub level_x: i32,
    pub level_y: i32,

    /// The actual width in a tile.
    pub width: i32,
    /// The actual height in a tile.
    pub height: i32,

    /// The tile's values, channel by channel, each `tile_size_x` by `tile_size_y` of the
    /// header (of which `width` by `height` are the tile's).
    pub images: Vec<ChannelData>,
}

/// `EXRImage`: the pixels of a file: channel by channel for a file of scanlines, tile by
/// tile for a tiled one, whose further levels follow in `next_level`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExrImage {
    /// Tiled pixel data: the tiles of this level.
    pub tiles: Vec<ExrTile>,
    /// The next level of a tiled image with mip or rip levels.
    pub next_level: Option<Box<ExrImage>>,
    pub level_x: i32,
    pub level_y: i32,

    /// The values of each channel, for a file of scanlines: `width` by `height`.
    pub images: Vec<ChannelData>,

    pub width: i32,
    pub height: i32,
    pub num_channels: i32,
}

/// `ParseEXRVersionFromMemory`: the eight bytes a file starts with. The error is the
/// original's code.
pub fn parse_exr_version_from_memory(memory: &[u8]) -> Result<ExrVersion, i32> {
    if memory.len() < 8 {
        return Err(ERROR_INVALID_DATA);
    }

    // Header check.
    if memory[..4] != [0x76, 0x2f, 0x31, 0x01] {
        return Err(ERROR_INVALID_MAGIC_NUMBER);
    }

    // version, scanline: must be 2.
    if memory[4] != 2 {
        return Err(ERROR_INVALID_EXR_VERSION);
    }

    let flags = memory[5];
    Ok(ExrVersion {
        version: 2,
        tiled: flags & 0x2 != 0,
        long_name: flags & 0x4 != 0,
        non_image: flags & 0x8 != 0,
        multipart: flags & 0x10 != 0,
    })
}

/// `IsEXRFromMemory`.
pub fn is_exr_from_memory(memory: &[u8]) -> bool {
    parse_exr_version_from_memory(memory).is_ok()
}

/// `ConvertHeader`: a parsed header as an `ExrHeader`. `None`, with the reason added to
/// `err`, where its type attribute and its tiles disagree.
pub(crate) fn convert_header(
    info: &HeaderInfo,
    warn: &mut String,
    err: &mut String,
) -> Option<ExrHeader> {
    let mut exr_header = ExrHeader {
        pixel_aspect_ratio: info.pixel_aspect_ratio,
        screen_window_center: info.screen_window_center,
        screen_window_width: info.screen_window_width,
        chunk_count: info.chunk_count,
        display_window: info.display_window,
        data_window: info.data_window,
        line_order: info.line_order,
        compression_type: info.compression_type,
        tiled: info.tiled,
        tile_size_x: info.tile_size_x,
        tile_size_y: info.tile_size_y,
        tile_level_mode: info.tile_level_mode,
        tile_rounding_mode: info.tile_rounding_mode,
        // `EXRSetNameAttr`: up to 255 bytes of it.
        name: bytes_to_string(&info.name.as_bytes()[..info.name.len().min(255)]),
        ..ExrHeader::default()
    };

    if !info.type_name.is_empty() {
        let mut valid = true;
        if info.type_name == "scanlineimage" {
            if exr_header.tiled != 0 {
                err.push_str("(ConvertHeader) tiled bit must be off for `scanlineimage` type.\n");
                valid = false;
            }
        } else if info.type_name == "tiledimage" {
            if exr_header.tiled == 0 {
                err.push_str("(ConvertHeader) tiled bit must be on for `tiledimage` type.\n");
                valid = false;
            }
        } else if info.type_name == "deeptile" {
            exr_header.non_image = 1;
            if exr_header.tiled == 0 {
                err.push_str("(ConvertHeader) tiled bit must be on for `deeptile` type.\n");
                valid = false;
            }
        } else if info.type_name == "deepscanline" {
            exr_header.non_image = 1;
            if exr_header.tiled != 0 {
                err.push_str("(ConvertHeader) tiled bit must be off for `deepscanline` type.\n");
                // valid = false;
            }
        } else {
            warn.push_str(&format!(
                "(ConvertHeader) Unsupported or unknown info.type: {}\n",
                info.type_name
            ));
        }

        if !valid {
            return None;
        }
    }

    exr_header.channels = info
        .channels
        .iter()
        .map(|channel| ChannelInfo {
            name: bytes_to_string(&channel.name.as_bytes()[..channel.name.len().min(255)]),
            ..channel.clone()
        })
        .collect();

    exr_header.pixel_types = info
        .channels
        .iter()
        .map(|channel| channel.pixel_type)
        .collect();

    // Initially fill with values of `pixel_types`.
    exr_header.requested_pixel_types = exr_header.pixel_types.clone();

    exr_header.custom_attributes = info
        .attributes
        .iter()
        .take(MAX_CUSTOM_ATTRIBUTES)
        .cloned()
        .collect();

    exr_header.header_len = info.header_len;

    Some(exr_header)
}

/// `ParseEXRHeaderFromMemory`: the header of a file of one part.
pub fn parse_exr_header_from_memory(
    version: &ExrVersion,
    memory: &[u8],
) -> Result<ExrHeader, ExrError> {
    parse_header(version, memory, false)
}

/// `ParseEXRHeaderFromMemory` exactly as tinyexr has it, which refuses DWAA and DWAB
/// ("Unknown compression type.").
pub fn parse_exr_header_from_memory_as_tinyexr(
    version: &ExrVersion,
    memory: &[u8],
) -> Result<ExrHeader, ExrError> {
    parse_header(version, memory, true)
}

fn parse_header(
    version: &ExrVersion,
    memory: &[u8],
    as_tinyexr: bool,
) -> Result<ExrHeader, ExrError> {
    let (exr_header, code, message) =
        parse_exr_header_from_memory_whole(version, memory, as_tinyexr);
    if code != 0 {
        return Err(ExrError { code, message });
    }
    Ok(exr_header)
}

/// `ParseEXRHeaderFromMemory` with all it answers: the original fills in the header as far
/// as it got even when it then says no.
fn parse_exr_header_from_memory_whole(
    version: &ExrVersion,
    memory: &[u8],
    as_tinyexr: bool,
) -> (ExrHeader, i32, String) {
    let mut err = String::new();
    if memory.len() < 8 {
        return (
            ExrHeader::default(),
            ERROR_INVALID_DATA,
            "Insufficient header/data size.\n".to_string(),
        );
    }

    let marker = &memory[8..];

    let mut info = HeaderInfo::default();

    let mut ret = 0;
    {
        let mut err_str = String::new();
        if let Err(code) = parse_exr_header(
            &mut info,
            &mut false,
            version,
            &mut err_str,
            marker,
            as_tinyexr,
        ) {
            ret = code;
            if !err_str.is_empty() {
                err = err_str;
            }
        }
    }

    let mut exr_header;
    {
        let mut warn = String::new();
        let mut err_str = String::new();

        match convert_header(&info, &mut warn, &mut err_str) {
            Some(header) => exr_header = header,
            None => {
                // (The original leaves the header with what it had filled in up to there.)
                exr_header = ExrHeader::default();
                if !err_str.is_empty() {
                    err = err_str;
                }
                ret = ERROR_INVALID_HEADER;
            }
        }
    }

    exr_header.multipart = i32::from(version.multipart);
    exr_header.non_image = i32::from(version.non_image);

    (exr_header, ret, err)
}

/// `LoadEXRImageFromMemory`: the pixels of the file whose header `exr_header` is. Each
/// channel comes as the type `exr_header.requested_pixel_types` asks for; afterwards
/// `exr_header.pixel_types` says the same.
///
/// A PXR24, B44 or B44A block that is as long as its pixels is read as the pixels, which is
/// the format's rule and not tinyexr's: see [`load_exr_image_from_memory_as_tinyexr`].
pub fn load_exr_image_from_memory(
    exr_header: &mut ExrHeader,
    memory: &[u8],
) -> Result<ExrImage, ExrError> {
    load_image(exr_header, memory, false)
}

/// `LoadEXRImageFromMemory` exactly as tinyexr has it, for every file, which
/// [`load_exr_image_from_memory`] is but for two kinds of block:
///
/// - tinyexr reads a B44 or B44A block as B44 whatever its length. The format has a block
///   that would not come out smaller stored as its pixels, line after line, and OpenEXR
///   writes it so: always, for an image whose channels are all floats or integers, which
///   B44 leaves as they are. tinyexr reads such a file as other values.
/// - tinyexr takes a PXR24 block as long as its values in PXR24's own layout for one that
///   was not compressed, which no writer makes; a compressed block that happens to come out
///   that long is read as other values, including one tinyexr wrote itself.
pub fn load_exr_image_from_memory_as_tinyexr(
    exr_header: &mut ExrHeader,
    memory: &[u8],
) -> Result<ExrImage, ExrError> {
    load_image(exr_header, memory, true)
}

fn load_image(
    exr_header: &mut ExrHeader,
    memory: &[u8],
    as_tinyexr: bool,
) -> Result<ExrImage, ExrError> {
    if memory.len() < 8 {
        return Err(ExrError {
            code: ERROR_INVALID_ARGUMENT,
            message: "Invalid argument for LoadEXRImageFromMemory".to_string(),
        });
    }

    if exr_header.header_len == 0 {
        return Err(ExrError {
            code: ERROR_INVALID_ARGUMENT,
            message: "EXRHeader variable is not initialized.".to_string(),
        });
    }

    let marker = exr_header.header_len as usize + 8; // +8 for magic number + version header.
    let mut exr_image = ExrImage::default();
    let mut err = String::new();
    let ret = chunk::decode_exr_image(
        &mut exr_image,
        exr_header,
        memory,
        marker,
        &mut err,
        as_tinyexr,
    );
    if ret != 0 {
        return Err(ExrError {
            code: ret,
            message: err,
        });
    }
    Ok(exr_image)
}

/// A value of a channel as a float, as the original reads every channel's memory.
#[inline]
fn value_as_float(image: &ChannelData, index: usize) -> f32 {
    match image {
        ChannelData::Float(values) => values[index],
        ChannelData::Uint(values) => f32::from_bits(values[index]),
        // (Not asked for by `load_exr_from_memory`, which reads halves as floats.)
        ChannelData::Half(values) => half_to_float(values[index]),
    }
}

/// `LoadEXRFromMemory`: a file as floats of red, green, blue and alpha, with its width and
/// height. A file of one channel gives that channel four times; any other needs channels
/// named `R`, `G` and `B`, and its alpha is 1 where it has no `A`.
pub fn load_exr_from_memory(memory: &[u8]) -> Result<(Vec<f32>, i32, i32), ExrError> {
    load_rgba(memory, false)
}

/// `LoadEXRFromMemory` exactly as tinyexr has it
/// ([`load_exr_image_from_memory_as_tinyexr`] says where that differs).
pub fn load_exr_from_memory_as_tinyexr(memory: &[u8]) -> Result<(Vec<f32>, i32, i32), ExrError> {
    load_rgba(memory, true)
}

fn load_rgba(memory: &[u8], as_tinyexr: bool) -> Result<(Vec<f32>, i32, i32), ExrError> {
    let exr_version = match parse_exr_version_from_memory(memory) {
        Ok(version) => version,
        Err(code) => {
            return Err(ExrError {
                code,
                message: format!("Failed to parse EXR version. code({code})"),
            });
        }
    };

    let mut exr_header = parse_header(&exr_version, memory, as_tinyexr)?;

    // Read HALF channel as FLOAT.
    for i in 0..exr_header.channels.len() {
        if exr_header.pixel_types[i] == PIXELTYPE_HALF {
            exr_header.requested_pixel_types[i] = PIXELTYPE_FLOAT;
        }
    }

    let exr_image = load_image(&mut exr_header, memory, as_tinyexr)?;

    // RGBA
    let mut idx_r = -1i32;
    let mut idx_g = -1i32;
    let mut idx_b = -1i32;
    let mut idx_a = -1i32;
    for (c, channel) in exr_header.channels.iter().enumerate() {
        match channel.name.as_str() {
            "R" => idx_r = c as i32,
            "G" => idx_g = c as i32,
            "B" => idx_b = c as i32,
            "A" => idx_a = c as i32,
            _ => {}
        }
    }

    let single = exr_header.channels.len() == 1;
    if !single {
        // Assume RGB(A)
        if idx_r == -1 {
            return Err(ExrError {
                code: ERROR_INVALID_DATA,
                message: "R channel not found".to_string(),
            });
        }

        if idx_g == -1 {
            return Err(ExrError {
                code: ERROR_INVALID_DATA,
                message: "G channel not found".to_string(),
            });
        }

        if idx_b == -1 {
            return Err(ExrError {
                code: ERROR_INVALID_DATA,
                message: "B channel not found".to_string(),
            });
        }
    }
    // Grayscale channel only: all four from the one channel.
    let sources: [Option<usize>; 4] = if single {
        [Some(0); 4]
    } else {
        [
            Some(idx_r as usize),
            Some(idx_g as usize),
            Some(idx_b as usize),
            if idx_a != -1 {
                Some(idx_a as usize)
            } else {
                None
            },
        ]
    };

    let (image_width, image_height) = (
        exr_image.width.max(0) as usize,
        exr_image.height.max(0) as usize,
    );
    // (The original's memory starts as whatever it holds; a place no tile covers is 0 here.)
    let mut out_rgba = vec![0f32; 4 * image_width * image_height];

    if exr_header.tiled != 0 {
        let tile_size_x = exr_header.tile_size_x as usize;
        let tile_size_y = exr_header.tile_size_y as usize;
        for tile in &exr_image.tiles {
            for j in 0..tile_size_y {
                for i in 0..tile_size_x {
                    let ii = (tile.offset_x as isize as usize)
                        .wrapping_mul(tile_size_x)
                        .wrapping_add(i);
                    let jj = (tile.offset_y as isize as usize)
                        .wrapping_mul(tile_size_y)
                        .wrapping_add(j);

                    // out of region check.
                    if ii >= image_width {
                        continue;
                    }
                    if jj >= image_height {
                        continue;
                    }
                    let idx = ii + jj * image_width;
                    let src_idx = i + j * tile_size_x;
                    for (k, source) in sources.iter().enumerate() {
                        out_rgba[4 * idx + k] = match source {
                            Some(c) => value_as_float(&tile.images[*c], src_idx),
                            None => 1.0,
                        };
                    }
                }
            }
        }
    } else {
        for i in 0..image_width * image_height {
            for (k, source) in sources.iter().enumerate() {
                out_rgba[4 * i + k] = match source {
                    Some(c) => value_as_float(&exr_image.images[*c], i),
                    None => 1.0,
                };
            }
        }
    }

    Ok((out_rgba, exr_image.width, exr_image.height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attribute(file: &mut Vec<u8>, name: &str, type_name: &str, value: &[u8]) {
        file.extend_from_slice(name.as_bytes());
        file.push(0);
        file.extend_from_slice(type_name.as_bytes());
        file.push(0);
        file.extend_from_slice(&(value.len() as u32).to_le_bytes());
        file.extend_from_slice(value);
    }

    /// A file of two by two pixels put together by hand: uncompressed scanlines, with
    /// channels B, G and R of halves. `pixels` are red, green and blue as halves, line by
    /// line.
    fn file_by_hand(pixels: &[[u16; 3]; 4]) -> Vec<u8> {
        let mut file = vec![0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0];
        let mut channels = Vec::new();
        for name in ["B", "G", "R"] {
            channels.extend_from_slice(name.as_bytes());
            channels.push(0);
            channels.extend_from_slice(&PIXELTYPE_HALF.to_le_bytes());
            channels.extend_from_slice(&[0, 0, 0, 0]);
            channels.extend_from_slice(&1i32.to_le_bytes());
            channels.extend_from_slice(&1i32.to_le_bytes());
        }
        channels.push(0);
        let window: Vec<u8> = [0i32, 0, 1, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        attribute(&mut file, "channels", "chlist", &channels);
        attribute(&mut file, "compression", "compression", &[0]);
        attribute(&mut file, "dataWindow", "box2i", &window);
        attribute(&mut file, "displayWindow", "box2i", &window);
        attribute(&mut file, "lineOrder", "lineOrder", &[0]);
        attribute(&mut file, "pixelAspectRatio", "float", &1f32.to_le_bytes());
        attribute(&mut file, "screenWindowCenter", "v2f", &[0; 8]);
        attribute(&mut file, "screenWindowWidth", "float", &1f32.to_le_bytes());
        file.push(0);
        // The table of offsets: one block a line.
        let first = file.len() + 16;
        let block = 8 + 3 * 2 * 2;
        file.extend_from_slice(&(first as u64).to_le_bytes());
        file.extend_from_slice(&((first + block) as u64).to_le_bytes());
        for y in 0..2usize {
            file.extend_from_slice(&(y as i32).to_le_bytes());
            file.extend_from_slice(&12i32.to_le_bytes());
            // A line's channels one after the other, in the order of their names.
            for channel in [2, 1, 0] {
                for x in 0..2 {
                    file.extend_from_slice(&pixels[y * 2 + x][channel].to_le_bytes());
                }
            }
        }
        file
    }

    #[test]
    fn a_file_put_together_by_hand_is_read() {
        // 1, 0.5 and 2 as halves.
        let (one, half, two) = (0x3c00, 0x3800, 0x4000);
        let pixels = [[one, half, two], [0, 0, 0], [two, two, two], [half, one, 0]];
        let file = file_by_hand(&pixels);

        let (rgba, width, height) = load_exr_from_memory(&file).unwrap();
        assert_eq!((width, height), (2, 2));
        assert_eq!(
            rgba,
            [
                1.0, 0.5, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.5, 1.0, 0.0, 1.0
            ]
        );

        // And channel by channel, as it is stored.
        let version = parse_exr_version_from_memory(&file).unwrap();
        assert!(!version.tiled && !version.multipart && !version.non_image);
        let mut header = parse_exr_header_from_memory(&version, &file).unwrap();
        assert_eq!(
            header
                .channels
                .iter()
                .map(|channel| channel.name.as_str())
                .collect::<Vec<_>>(),
            ["B", "G", "R"]
        );
        assert_eq!(header.compression_type, COMPRESSIONTYPE_NONE);
        let image = load_exr_image_from_memory(&mut header, &file).unwrap();
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.images[2], ChannelData::Half(vec![one, 0, two, half]));
        assert_eq!(image.images[0], ChannelData::Half(vec![two, 0, two, 0]));
    }

    #[test]
    fn a_file_that_is_not_one_is_refused_with_tinyexrs_code() {
        assert_eq!(
            parse_exr_version_from_memory(b"\x89PNG\r\n\x1a\n"),
            Err(ERROR_INVALID_MAGIC_NUMBER)
        );
        assert_eq!(
            parse_exr_version_from_memory(&[0x76, 0x2f, 0x31]),
            Err(ERROR_INVALID_DATA)
        );
        let error = load_exr_from_memory(b"not an .exr file").unwrap_err();
        assert_eq!(error.code, ERROR_INVALID_MAGIC_NUMBER);
        assert_eq!(error.message, "Failed to parse EXR version. code(-1)");

        // Cut off within its last block.
        let file = file_by_hand(&[[0; 3]; 4]);
        let error = load_exr_from_memory(&file[..file.len() - 4]).unwrap_err();
        assert_eq!(error.code, ERROR_INVALID_DATA);
    }

    /// An image of scanlines, `width` by `height`, of channels B, G and R as `values` gives
    /// them, with the header to save it by.
    fn image_of(
        width: i32,
        height: i32,
        compression_type: i32,
        values: impl Fn(usize, usize) -> ChannelData,
        file_type: i32,
        memory_type: i32,
    ) -> (ExrImage, ExrHeader) {
        let image = ExrImage {
            images: (0..3)
                .map(|c| values(c, (width * height) as usize))
                .collect(),
            width,
            height,
            num_channels: 3,
            ..ExrImage::default()
        };
        let header = ExrHeader {
            channels: ["B", "G", "R"]
                .iter()
                .map(|name| ChannelInfo {
                    name: name.to_string(),
                    pixel_type: file_type,
                    x_sampling: 1,
                    y_sampling: 1,
                    p_linear: 0,
                })
                .collect(),
            pixel_types: vec![memory_type; 3],
            requested_pixel_types: vec![file_type; 3],
            compression_type,
            ..ExrHeader::default()
        };
        (image, header)
    }

    fn read(file: &[u8], as_tinyexr: bool) -> ExrImage {
        let version = parse_exr_version_from_memory(file).unwrap();
        let mut header = parse_exr_header_from_memory(&version, file).unwrap();
        if as_tinyexr {
            load_exr_image_from_memory_as_tinyexr(&mut header, file).unwrap()
        } else {
            load_exr_image_from_memory(&mut header, file).unwrap()
        }
    }

    #[test]
    fn dwa_is_written_and_read_but_not_as_tinyexr() {
        // A smooth image: the lossy compression keeps it near, and smaller.
        let smooth = |c: usize, count: usize| {
            ChannelData::Float(
                (0..count)
                    .map(|i| {
                        ((i % 70) as f32 * 0.01 + (i / 70) as f32 * 0.02 + c as f32 * 0.1)
                            .sin()
                            .abs()
                    })
                    .collect(),
            )
        };
        for compression_type in [COMPRESSIONTYPE_DWAA, COMPRESSIONTYPE_DWAB] {
            let (image, mut header) = image_of(
                70,
                300,
                compression_type,
                smooth,
                PIXELTYPE_HALF,
                PIXELTYPE_FLOAT,
            );
            let file = save_exr_image_to_memory(&image, &header).unwrap();
            let (rgba, width, height) = load_exr_from_memory(&file).unwrap();
            assert_eq!((width, height), (70, 300));
            let ChannelData::Float(red) = &image.images[2] else {
                unreachable!()
            };
            assert!(
                rgba.as_chunks::<4>()
                    .0
                    .iter()
                    .zip(red)
                    .all(|(pixel, red)| (pixel[0] - red).abs() < 0.02 && pixel[3] == 1.0),
                "compression {compression_type}"
            );
            let (_, plain) = image_of(
                70,
                300,
                COMPRESSIONTYPE_NONE,
                smooth,
                PIXELTYPE_HALF,
                PIXELTYPE_FLOAT,
            );
            assert!(file.len() * 3 < save_exr_image_to_memory(&image, &plain).unwrap().len());

            // tinyexr knows neither: it writes no such file, and reads none.
            assert!(save_exr_image_to_memory_as_tinyexr(&image, &header).is_err());
            let version = parse_exr_version_from_memory(&file).unwrap();
            let refused = parse_exr_header_from_memory_as_tinyexr(&version, &file).unwrap_err();
            assert_eq!(
                (refused.code, refused.message.as_str()),
                (ERROR_UNSUPPORTED_FORMAT, "Unknown compression type.")
            );
            assert!(load_exr_from_memory_as_tinyexr(&file).is_err());

            // The level of loss, as OpenEXR bounds it.
            header.custom_attributes = vec![dwa_compression_level_attribute(-1.0)];
            assert_eq!(
                save_exr_image_to_memory(&image, &header)
                    .unwrap_err()
                    .message,
                DWA_INVALID_LEVEL
            );
            header.custom_attributes = vec![dwa_compression_level_attribute(0.0)];
            let lossless_quantization = save_exr_image_to_memory(&image, &header).unwrap();
            assert!(lossless_quantization.len() > file.len());
        }
    }

    #[test]
    fn what_is_written_comes_back() {
        let floats = |c: usize, count: usize| {
            ChannelData::Float(
                (0..count)
                    .map(|i| ((i * 31 + c * 7) % 97) as f32 * 0.125 - 3.0)
                    .collect(),
            )
        };
        for compression_type in [
            COMPRESSIONTYPE_NONE,
            COMPRESSIONTYPE_RLE,
            COMPRESSIONTYPE_ZIPS,
            COMPRESSIONTYPE_ZIP,
            COMPRESSIONTYPE_PIZ,
            COMPRESSIONTYPE_B44,
            COMPRESSIONTYPE_B44A,
        ] {
            // (B44 leaves floats as they are.)
            let (image, header) = image_of(
                21,
                40,
                compression_type,
                floats,
                PIXELTYPE_FLOAT,
                PIXELTYPE_FLOAT,
            );
            let file = save_exr_image_to_memory(&image, &header).unwrap();
            assert_eq!(
                read(&file, false).images,
                image.images,
                "compression {compression_type}"
            );
            // And as halves, which these values all are.
            let (_, header) = image_of(
                21,
                40,
                compression_type,
                floats,
                PIXELTYPE_HALF,
                PIXELTYPE_FLOAT,
            );
            let file = save_exr_image_to_memory(&image, &header).unwrap();
            let (rgba, width, height) = load_exr_from_memory(&file).unwrap();
            assert_eq!((width, height), (21, 40));
            if compression_type != COMPRESSIONTYPE_B44 && compression_type != COMPRESSIONTYPE_B44A {
                let ChannelData::Float(red) = &image.images[2] else {
                    unreachable!()
                };
                assert!(
                    rgba.as_chunks::<4>()
                        .0
                        .iter()
                        .zip(red)
                        .all(|(pixel, red)| pixel[0] == *red && pixel[3] == 1.0),
                    "compression {compression_type}"
                );
            }
        }
        // PXR24 keeps integers and halves whole.
        let halves = |c: usize, count: usize| {
            ChannelData::Half((0..count).map(|i| (i * 977 + c * 13) as u16).collect())
        };
        let (image, header) = image_of(
            21,
            40,
            COMPRESSIONTYPE_PXR24,
            halves,
            PIXELTYPE_HALF,
            PIXELTYPE_HALF,
        );
        assert_eq!(
            read(&save_exr_image_to_memory(&image, &header).unwrap(), false).images,
            image.images
        );
    }

    #[test]
    fn a_block_that_does_not_come_out_smaller_is_stored_as_it_is() {
        // B44 leaves floats as they are, so no block of an image of floats comes out smaller:
        // the format has such a block stored as its pixels, line after line, which is what
        // a file that is not compressed holds. tinyexr stores it channel after channel.
        let floats = |c: usize, count: usize| {
            ChannelData::Float((0..count).map(|i| (i * 3 + c) as f32).collect())
        };
        let (image, header) = image_of(
            5,
            3,
            COMPRESSIONTYPE_B44,
            floats,
            PIXELTYPE_FLOAT,
            PIXELTYPE_FLOAT,
        );
        let file = save_exr_image_to_memory(&image, &header).unwrap();
        let as_tinyexr = save_exr_image_to_memory_as_tinyexr(&image, &header).unwrap();
        assert_eq!(file.len(), as_tinyexr.len());
        assert_ne!(file, as_tinyexr);
        let (_, plain_header) = image_of(
            5,
            3,
            COMPRESSIONTYPE_NONE,
            floats,
            PIXELTYPE_FLOAT,
            PIXELTYPE_FLOAT,
        );
        let plain = save_exr_image_to_memory(&image, &plain_header).unwrap();
        // (A B44 block holds the three lines and a plain one a line, each 60 bytes after the
        // 8 the block starts with: the first line's pixels are the same bytes in both.)
        assert_eq!(
            file[file.len() - 3 * 60..file.len() - 2 * 60],
            plain[plain.len() - 3 * 68 + 8..plain.len() - 2 * 68]
        );
        // Each way reads its own, and not the other's.
        assert_eq!(read(&file, false).images, image.images);
        assert_eq!(read(&as_tinyexr, true).images, image.images);
        assert_ne!(read(&as_tinyexr, false).images, image.images);
        assert_ne!(read(&file, true).images, image.images);

        // A PXR24 block of two pixels is longer compressed than not.
        let halves = |c: usize, count: usize| {
            ChannelData::Half((0..count).map(|i| (i * 977 + c * 13 + 5) as u16).collect())
        };
        let (image, header) = image_of(
            2,
            1,
            COMPRESSIONTYPE_PXR24,
            halves,
            PIXELTYPE_HALF,
            PIXELTYPE_HALF,
        );
        let file = save_exr_image_to_memory(&image, &header).unwrap();
        let as_tinyexr = save_exr_image_to_memory_as_tinyexr(&image, &header).unwrap();
        assert!(file.len() < as_tinyexr.len());
        assert_eq!(read(&file, false).images, image.images);
        assert_eq!(read(&as_tinyexr, true).images, image.images);
        // (Longer than its pixels, a block is no longer taken for them.)
        assert_eq!(read(&as_tinyexr, false).images, image.images);
    }

    #[test]
    fn what_the_original_would_read_past_is_refused() {
        let floats = |_: usize, count: usize| ChannelData::Float(vec![0.5; count]);
        let (image, header) = image_of(
            4,
            4,
            COMPRESSIONTYPE_ZIP,
            floats,
            PIXELTYPE_FLOAT,
            PIXELTYPE_FLOAT,
        );
        assert!(save_exr_image_to_memory(&image, &header).is_ok());
        // A channel short of a value; a type that is not the values'; no size.
        let mut short = image.clone();
        short.images[1] = ChannelData::Float(vec![0.5; 15]);
        assert_eq!(
            save_exr_image_to_memory(&short, &header).unwrap_err().code,
            ERROR_INVALID_ARGUMENT
        );
        let mut other = header.clone();
        other.pixel_types[0] = PIXELTYPE_HALF;
        assert_eq!(
            save_exr_image_to_memory(&image, &other).unwrap_err().code,
            ERROR_INVALID_ARGUMENT
        );
        let mut flat = image.clone();
        flat.height = 0;
        assert_eq!(
            save_exr_image_to_memory(&flat, &header).unwrap_err().code,
            ERROR_INVALID_ARGUMENT
        );
        // ZFP of a block that is not of whole blocks of four by four.
        let (odd, mut zfp) = image_of(
            6,
            4,
            COMPRESSIONTYPE_ZFP,
            floats,
            PIXELTYPE_FLOAT,
            PIXELTYPE_FLOAT,
        );
        assert_eq!(
            save_exr_image_to_memory(&odd, &zfp).unwrap_err().message,
            ZFP_NOT_MULTIPLE_OF_FOUR
        );
        // And what the original refuses itself, with its message.
        zfp.requested_pixel_types[1] = PIXELTYPE_HALF;
        let error = save_exr_image_to_memory(&odd, &zfp).unwrap_err();
        assert_eq!(
            (error.code, error.message.as_str()),
            (
                ERROR_SERIALIZATION_FAILED,
                "Pixel type must be FLOAT for ZFP compression"
            )
        );
        let error = save_exr_to_memory(&[0.0; 8], 2, 2, 2, false).unwrap_err();
        let _ = &image;
        assert_eq!(
            (error.code, error.message.as_str()),
            (ERROR_INVALID_ARGUMENT, "Unsupported component value : 2\n")
        );
    }

    #[test]
    fn the_levels_of_a_tiled_image() {
        let mut header = ExrHeader {
            data_window: Box2i {
                min_x: 0,
                min_y: 0,
                max_x: 9,
                max_y: 4,
            },
            tile_size_x: 4,
            tile_size_y: 4,
            tile_level_mode: TILE_ONE_LEVEL,
            ..ExrHeader::default()
        };
        assert_eq!(
            tile_levels(&header).unwrap(),
            [TileLevel {
                level_x: 0,
                level_y: 0,
                width: 10,
                height: 5,
                num_x_tiles: 3,
                num_y_tiles: 2
            }]
        );
        // Ten pixels halve to five, two and one, rounded down; or to five, three, two and
        // one, rounded up.
        header.tile_level_mode = TILE_MIPMAP_LEVELS;
        let sizes = |header: &ExrHeader| {
            tile_levels(header)
                .unwrap()
                .iter()
                .map(|level| (level.width, level.height))
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(&header), [(10, 5), (5, 2), (2, 1), (1, 1)]);
        header.tile_rounding_mode = TILE_ROUND_UP;
        assert_eq!(sizes(&header), [(10, 5), (5, 3), (3, 2), (2, 1), (1, 1)]);
        // Rip levels: every width with every height, row after row.
        header.tile_level_mode = TILE_RIPMAP_LEVELS;
        header.tile_rounding_mode = TILE_ROUND_DOWN;
        let levels = tile_levels(&header).unwrap();
        assert_eq!(levels.len(), 4 * 3);
        assert_eq!(
            (
                levels[1].level_x,
                levels[1].level_y,
                levels[1].width,
                levels[1].height
            ),
            (1, 0, 5, 5)
        );
        assert_eq!(
            (
                levels[11].level_x,
                levels[11].level_y,
                levels[11].width,
                levels[11].height
            ),
            (3, 2, 1, 1)
        );
        // No tile size, no levels.
        header.tile_size_x = 0;
        assert!(tile_levels(&header).is_none());
    }

    #[test]
    fn halves_and_floats() {
        for (half, float) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x0000, 0.0),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ] {
            assert_eq!(half_to_float(half), float);
            assert_eq!(float_to_half_full(float), half);
        }
        assert!(half_to_float(0x7e00).is_nan());
        // Too large for a half.
        assert_eq!(float_to_half_full(1e10), 0x7c00);
    }
}

/// (The README's examples, compiled and run with the tests.)
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
