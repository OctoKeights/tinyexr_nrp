//! Deep images: a port of `LoadDeepEXR` and `DecodeDeepTiled` of tinyexr v3.4.0
//! (`tinyexr.h`), read from memory. tinyexr reads deep images and does not write them.
//!
//! What tinyexr reads, and so the port:
//!
//! - **Deep scanlines** whose blocks are ZIPS-compressed (one line a block). The reader takes
//!   each block for one line, so with ZIP (sixteen lines a block) only the first of every
//!   sixteen lines has samples, and the others none. A block it cannot decompress ends the
//!   reading there and is not reported: the image comes back empty, with no width, height,
//!   channels or samples (the original's answer is success, with a width and height of 0 and
//!   the lines it had made so far, which a reader of such an image does not look at).
//! - **Deep tiles** of one level, uncompressed or compressed with RLE, ZIPS or ZIP.
//!
//! The original reads past the end of a damaged file, and before its start, without a look:
//! for the table of offsets of a file of scanlines and for each of its blocks. Here what is
//! past the end is taken to be all zeros and then all ones, as in [`super::parts`]; where the
//! answer differs, or the read is before the file or far past its end, the file is refused
//! with [`super::parts::PAST_THE_END`]. So is a file the original stops on (an attribute too
//! short for what it reads of it, a size it cannot allocate).

use crate::TileLevelMode;
use std::borrow::Cow;
use std::cell::Cell;

use super::parts::PAST_THE_END;
use super::zip_rle::{decompress_rle, decompress_zip, unpredict_and_reorder};
use super::{
    ChannelInfo, Compression, ErrorCode, ExrError, PixelType, half_to_float, read_attribute,
    read_channel_info,
};
use miniz_nrp::{DecompressError, Format};

/// `DeepImage`: the samples of a deep image, line by line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeepImage {
    /// The name of each channel.
    pub channel_names: Vec<String>,
    /// `image[channel][line]`: the samples of a line, every pixel's one after another. (Of a
    /// file of scanlines, the lines that have a block of their own: see the module's notes.)
    pub image: Vec<Vec<Vec<f32>>>,
    /// `offset_table[line][x]`: how many samples the line has up to and including pixel `x`.
    pub offset_table: Vec<Vec<i32>>,
    /// How many channels it has.
    pub num_channels: i32,
    /// Its width in pixels.
    pub width: i32,
    /// Its height in pixels.
    pub height: i32,
}

/// How far past the end of a file a read is still taken as reading what is there: 64 KiB,
/// and on to the next multiple of 4 KiB from the start of the file. (Where the comparison
/// with the original puts memory it cannot read.)
fn reach(size: u64) -> u64 {
    (size + 65536).div_ceil(4096) * 4096
}

/// The largest number of pixels a deep image is read with: 67 million. (The original
/// allocates a count for each, however many.)
const MAX_PIXELS: u64 = 1 << 26;

/// The most bytes miniz's `mz_uncompress` takes in: past it, it fails at once.
const MZ_LIMIT: u64 = 0xFFFF_FFFF;

/// The largest block of a file of scanlines that is decompressed: 1 GiB. (The original
/// allocates what a block says it comes to twice over before it looks at the block, and a
/// file that has it allocate more is one it may stop on, for want of memory: it is refused.)
const MAX_BLOCK: u64 = 1 << 30;

/// The bytes of the file as the original reads them, with `pad` past its end.
struct Memory<'a> {
    bytes: &'a [u8],
    pad: u8,
    beyond: Cell<bool>,
}

/// Why the reading stopped: an error of the original's, or a read it cannot make.
enum Stop {
    Error(ExrError),
    Outside,
}

impl From<ExrError> for Stop {
    fn from(error: ExrError) -> Stop {
        Stop::Error(error)
    }
}

fn error(code: ErrorCode, message: &str) -> Stop {
    Stop::Error(ExrError {
        code,
        message: message.to_string(),
    })
}

impl<'a> Memory<'a> {
    /// `len` bytes from `at`: past the end, the pad.
    fn read(&self, at: i64, len: u64) -> Result<Cow<'a, [u8]>, Stop> {
        let size = self.bytes.len() as u64;
        if at < 0 {
            return Err(Stop::Outside);
        }
        let at = at as u64;
        let end = at.checked_add(len).ok_or(Stop::Outside)?;
        if end <= size {
            return Ok(Cow::Borrowed(&self.bytes[at as usize..end as usize]));
        }
        if end > reach(size) {
            return Err(Stop::Outside);
        }
        self.beyond.set(true);
        Ok(Cow::Owned(
            (at..end)
                .map(|i| {
                    if i < size {
                        self.bytes[i as usize]
                    } else {
                        self.pad
                    }
                })
                .collect(),
        ))
    }

    /// Up to `len` bytes from `at`, as many as there are up to [`reach`]: for a
    /// decompressor, which reads only as far as its stream goes.
    fn read_up_to(&self, at: i64, len: u64) -> Result<Cow<'a, [u8]>, Stop> {
        let size = self.bytes.len() as u64;
        if at < 0 || at as u64 > reach(size) {
            return Err(Stop::Outside);
        }
        let len = len.min(reach(size) - at as u64);
        self.read(at, len)
    }

    fn i32_at(&self, at: i64) -> Result<i32, Stop> {
        Ok(i32::from_le_bytes(
            self.read(at, 4)?[..4].try_into().unwrap(),
        ))
    }

    fn i64_at(&self, at: i64) -> Result<i64, Stop> {
        Ok(i64::from_le_bytes(
            self.read(at, 8)?[..8].try_into().unwrap(),
        ))
    }
}

/// `LoadDeepEXR`, of a file in memory: its deep image.
///
/// # Errors
///
/// A file that is not a deep image, or that is damaged, with tinyexr's error code and message.
pub fn load_deep(memory: &[u8]) -> Result<DeepImage, ExrError> {
    let zeros = Memory {
        bytes: memory,
        pad: 0x00,
        beyond: Cell::new(false),
    };
    let first = load(&zeros);
    if let Err(Stop::Outside) = first {
        return Err(ExrError {
            code: ErrorCode::INVALID_DATA,
            message: PAST_THE_END.to_string(),
        });
    }
    if !zeros.beyond.get() {
        return first.map_err(|stop| match stop {
            Stop::Error(error) => error,
            Stop::Outside => unreachable!(),
        });
    }
    let ones = Memory {
        bytes: memory,
        pad: 0xff,
        beyond: Cell::new(false),
    };
    let second = load(&ones);
    match (first, second) {
        (Ok(a), Ok(b)) if same(&a, &b) => Ok(a),
        (Err(Stop::Error(a)), Err(Stop::Error(b))) if a == b => Err(a),
        _ => Err(ExrError {
            code: ErrorCode::INVALID_DATA,
            message: PAST_THE_END.to_string(),
        }),
    }
}

/// Whether two images are the same, their samples bit for bit.
fn same(a: &DeepImage, b: &DeepImage) -> bool {
    a.channel_names == b.channel_names
        && a.offset_table == b.offset_table
        && (a.num_channels, a.width, a.height) == (b.num_channels, b.width, b.height)
        && a.image.len() == b.image.len()
        && a.image.iter().zip(&b.image).all(|(a, b)| {
            a.len() == b.len()
                && a.iter().zip(b).all(|(a, b)| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                })
        })
}

/// `LoadDeepEXR` itself.
// (One pass, step for step as tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_lines)]
fn load(memory: &Memory) -> Result<DeepImage, Stop> {
    let file_size = memory.bytes.len();
    if file_size == 0 {
        return Err(error(ErrorCode::INVALID_FILE, "File size is zero"));
    }

    // Header check.
    if memory.read(0, 4)?[..] != [0x76, 0x2f, 0x31, 0x01] {
        return Err(error(
            ErrorCode::INVALID_MAGIC_NUMBER,
            "Invalid magic number",
        ));
    }
    let mut marker = 4i64;

    // Version, scanline.
    {
        // ver 2.0, scanline, deep bit on(0x800)
        // must be [2, 0, 0, 0]
        let version = memory.read(marker, 4)?;
        if version[0] != 2 || version[1] != 8 || version[2] != 0 || version[3] != 0 {
            return Err(error(
                ErrorCode::UNSUPPORTED_FORMAT,
                "Unsupported version or scanline",
            ));
        }
        marker += 4;
    }

    let mut dx = -1i32;
    let mut dy = -1i32;
    let mut dw = -1i32;
    let mut dh = -1i32;
    let mut num_scanline_blocks = 1i32; // 16 for ZIP compression.
    let mut compression_type = Compression(-1);
    let mut num_channels = -1i32;
    let mut is_tiled = false; // deep *tiled* part (type == "deeptile")
    let mut tile_size_x = -1i32;
    let mut tile_size_y = -1i32;
    let mut tile_level_mode = 0i32; // 0 = ONE_LEVEL
    let mut channels: Vec<ChannelInfo> = Vec::new();

    // Read attributes
    // (A file shorter than its version leaves the original a size below zero, and reading on.)
    let mut size = (file_size as u64).wrapping_sub(8);
    loop {
        if size == 0 {
            return Err(error(ErrorCode::INVALID_DATA, ""));
        } else if memory.read(marker, 1)?[0] == 0 {
            marker += 1;
            break;
        }

        let block = memory.read(marker, size)?;
        let Some((attr_name, attr_type, data, marker_size)) = read_attribute(&block) else {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Failed to parse attribute\n",
            ));
        };
        let _ = attr_type;
        marker += marker_size as i64;
        size -= marker_size as u64;

        // (What the original reads of an attribute too short for it is not there: it stops, or
        // reads past the attribute.)
        let need = |len: usize| {
            if data.len() < len {
                Err(Stop::Outside)
            } else {
                Ok(())
            }
        };
        let int_at = |at: usize| i32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        match attr_name.as_str() {
            "compression" => {
                need(1)?;
                compression_type = Compression(i32::from(data[0]));
                if compression_type > Compression::B44A {
                    return Err(Stop::Error(ExrError {
                        code: ErrorCode::UNSUPPORTED_FORMAT,
                        message: format!("Unsupported compression type : {}", compression_type.0),
                    }));
                }

                if compression_type == Compression::ZIP {
                    num_scanline_blocks = 16;
                } else if compression_type == Compression::B44
                    || compression_type == Compression::B44A
                {
                    num_scanline_blocks = 32;
                }
            }
            "channels" => {
                // name: zero-terminated string, from 1 to 255 bytes long
                // pixel type: int, possible values are: UINT = 0 HALF = 1 FLOAT = 2
                // pLinear: unsigned char, possible values are 0 and 1
                // reserved: three chars, should be zero
                // xSampling: int
                // ySampling: int
                if !read_channel_info(&mut channels, &data) {
                    return Err(error(
                        ErrorCode::INVALID_DATA,
                        "Failed to parse channel info",
                    ));
                }

                num_channels = channels.len() as i32;

                if num_channels < 1 {
                    return Err(error(ErrorCode::INVALID_DATA, "Invalid channels format"));
                }
            }
            "dataWindow" => {
                need(16)?;
                dx = int_at(0);
                dy = int_at(4);
                dw = int_at(8);
                dh = int_at(12);
            }
            "displayWindow" => {
                // (Its values are not used: only what the original stops on matters.)
                need(13)?;
            }
            "tiles" => {
                // Deep tiled part: parse the tile descriptor (xSize, ySize, mode).
                is_tiled = true;
                if data.len() >= 9 {
                    tile_size_x = int_at(0);
                    tile_size_y = int_at(4);
                    tile_level_mode = i32::from(data[8] & 0x3);
                }
            }
            "type" if data.len() >= 8 && &data[..8] == b"deeptile" => {
                is_tiled = true;
            }
            _ => {}
        }
    }

    if dx < 0 || dy < 0 || dw < 0 || dh < 0 || num_channels < 1 {
        return Err(error(ErrorCode::INVALID_DATA, ""));
    }

    let data_width = dw.wrapping_sub(dx).wrapping_add(1);
    let data_height = dh.wrapping_sub(dy).wrapping_add(1);

    if is_tiled {
        return decode_deep_tiled(
            memory,
            &channels,
            num_channels,
            compression_type,
            data_width,
            data_height,
            tile_size_x,
            tile_size_y,
            TileLevelMode(tile_level_mode),
            marker,
        );
    }

    // Read offset tables.
    let mut num_blocks = data_height / num_scanline_blocks;
    if num_blocks.wrapping_mul(num_scanline_blocks) < data_height {
        num_blocks += 1;
    }
    if num_blocks < 0 {
        return Err(Stop::Outside); // (a table the original cannot allocate)
    }

    let mut offsets = Vec::with_capacity(num_blocks.min(1 << 20) as usize);
    for _ in 0..num_blocks {
        offsets.push(memory.i64_at(marker)?);
        marker += 8;
    }

    if !(compression_type == Compression::NONE
        || compression_type == Compression::RLE
        || compression_type == Compression::ZIPS
        || compression_type == Compression::ZIP
        || compression_type == Compression::PIZ)
    {
        return Err(error(
            ErrorCode::UNSUPPORTED_FORMAT,
            "Unsupported compression format",
        ));
    }

    if num_blocks > 0 && data_width <= 0 {
        return Err(Stop::Outside); // (a line of no pixels, which the original stops on)
    }
    let (width, height) = (data_width.max(0) as usize, data_height.max(0) as usize);
    if width as u64 * height as u64 > MAX_PIXELS {
        return Err(Stop::Outside);
    }

    let mut deep_image = DeepImage {
        image: vec![Vec::new(); num_channels as usize],
        offset_table: vec![vec![0i32; width]; height],
        ..DeepImage::default()
    };

    for &offset in &offsets {
        let y = deep_image.image[0].len();

        // int: y coordinate
        // int64: packed size of pixel offset table
        // int64: packed size of sample data
        // int64: unpacked size of sample data
        // compressed pixel offset table
        // compressed sample data
        // (The line number is not used, and not read: built with optimisation, the original
        // does not read it either.)
        let packed_offset_table_size = memory.i64_at(offset.wrapping_add(4))?;
        let packed_sample_data_size = memory.i64_at(offset.wrapping_add(12))?;
        let unpacked_sample_data_size = memory.i64_at(offset.wrapping_add(20))?;

        let mut pixel_offset_table = vec![0i32; width];

        // decode pixel offset table.
        {
            let Some(table) = decompress_zip_from(
                memory,
                offset.wrapping_add(28),
                packed_offset_table_size as u64,
                width as u64 * 4,
            )?
            else {
                return Ok(unfinished()); // (the original's `return false`, which is 0)
            };

            if table.len() != width * 4 {
                return Err(error(ErrorCode::INVALID_DATA, ""));
            }
            for (value, bytes) in pixel_offset_table
                .iter_mut()
                .zip(table.as_chunks::<4>().0.iter())
            {
                *value = i32::from_le_bytes(*bytes);
            }
            deep_image.offset_table[y].copy_from_slice(&pixel_offset_table);
        }

        // (A size below zero is one the original cannot allocate, and one past `MAX_BLOCK` one
        // it may not be able to.)
        if unpacked_sample_data_size < 0 || unpacked_sample_data_size as u64 > MAX_BLOCK {
            return Err(Stop::Outside);
        }
        let mut sample_data = Vec::new();

        // decode sample data.
        if unpacked_sample_data_size != 0 {
            let Some(samples) = decompress_zip_from(
                memory,
                offset
                    .wrapping_add(28)
                    .wrapping_add(packed_offset_table_size),
                packed_sample_data_size as u64,
                unpacked_sample_data_size as u64,
            )?
            else {
                return Ok(unfinished());
            };
            if samples.len() as u64 != unpacked_sample_data_size as u64 {
                return Err(error(ErrorCode::INVALID_DATA, ""));
            }
            sample_data = samples;
        }

        // decode sample
        let mut channel_offset = 0i32;
        for channel in &channels {
            channel_offset += match channel.pixel_type {
                PixelType::UINT | PixelType::FLOAT => 4,
                PixelType::HALF => 2,
                _ => {
                    return Err(error(
                        ErrorCode::INVALID_DATA,
                        "Invalid pixel_type in chnnels.",
                    ));
                }
            };
        }
        let sample_size = channel_offset;
        if sample_size < 2 {
            return Err(error(ErrorCode::INVALID_DATA, ""));
        }

        if i64::from(pixel_offset_table[width - 1].wrapping_mul(sample_size)) as u64
            != sample_data.len() as u64
        {
            return Err(error(ErrorCode::INVALID_DATA, ""));
        }
        let samples_per_line = (sample_data.len() as i32 / sample_size) as usize;

        // pixel data is stored as image[channels][pixel_samples]
        let mut data_offset = 0usize;
        for (c, channel) in channels.iter().enumerate() {
            let line: Vec<f32> = match channel.pixel_type {
                PixelType::UINT => (0..samples_per_line)
                    .map(|x| {
                        u32::from_le_bytes(
                            sample_data[data_offset + x * 4..data_offset + x * 4 + 4]
                                .try_into()
                                .unwrap(),
                        ) as f32
                    })
                    .collect(), // @fixme
                PixelType::HALF => (0..samples_per_line)
                    .map(|x| {
                        half_to_float(u16::from_le_bytes(
                            sample_data[data_offset + x * 2..data_offset + x * 2 + 2]
                                .try_into()
                                .unwrap(),
                        ))
                    })
                    .collect(),
                _ => (0..samples_per_line)
                    .map(|x| {
                        f32::from_le_bytes(
                            sample_data[data_offset + x * 4..data_offset + x * 4 + 4]
                                .try_into()
                                .unwrap(),
                        )
                    })
                    .collect(),
            };
            data_offset += samples_per_line
                * if channel.pixel_type == PixelType::HALF {
                    2
                } else {
                    4
                };
            deep_image.image[c].push(line);
        }
    } // y

    deep_image.width = data_width;
    deep_image.height = data_height;
    deep_image.channel_names = channels
        .iter()
        .map(|channel| channel.name.clone())
        .collect();
    deep_image.num_channels = num_channels;

    Ok(deep_image)
}

/// What the original hands back when a block does not decompress, as far as a reader can
/// make use of it: an image of no width, height or channels. (The lines it had made before
/// are there, and not looked at; past the end of a damaged file they can be what is there.)
fn unfinished() -> DeepImage {
    DeepImage::default()
}

/// The most bytes `src_len` bytes can come to: stored as they are, or deflated (which makes
/// at most 1032 bytes of one), or run-length encoded (at most 128 of two).
fn most_from(src_len: usize) -> u64 {
    src_len as u64 * 1032 + 1024
}

/// `DecompressZip` of the `src_len` bytes at `at` into a block of `dst_len`, reading as far as
/// its stream goes: the bytes it came to (as many as the stream holds, up to `dst_len`), or
/// None where the original gives false.
fn decompress_zip_from(
    memory: &Memory,
    at: i64,
    src_len: u64,
    dst_len: u64,
) -> Result<Option<Vec<u8>>, Stop> {
    if dst_len == src_len {
        // Data is not compressed (Issue 40).
        return Ok(Some(memory.read(at, src_len)?.into_owned()));
    }
    // (A block of no bytes, which it stops on; or a larger one than it may be able to
    // allocate.)
    if dst_len == 0 || dst_len > MAX_BLOCK {
        return Err(Stop::Outside);
    }
    if src_len > MZ_LIMIT {
        return Ok(None); // (`mz_uncompress` fails without reading)
    }
    let src = memory.read_up_to(at, src_len)?;
    // (No more can come out of the stream than this, so no more need be made room for.)
    let room = dst_len.min(most_from(src.len())) as usize;
    match miniz_nrp::decompress_with(&src, Format::Zlib, room) {
        Ok(result) => {
            let mut tmp_buf = result.data;
            let mut dst = vec![0u8; tmp_buf.len()];
            unpredict_and_reorder(&mut dst, &mut tmp_buf);
            Ok(Some(dst))
        }
        // (A stream that runs on past what there is of it goes on past the end of the file. One
        // that fails before then fails there.)
        Err(DecompressError::Truncated) if (src.len() as u64) < src_len => Err(Stop::Outside),
        Err(_) => Ok(None),
    }
}

/// `DecompressDeep`
fn decompress_deep(compression_type: Compression, dst: &mut [u8], src: &[u8]) -> bool {
    if dst.is_empty() {
        return true;
    }
    if compression_type == Compression::RLE {
        return decompress_rle(dst, dst.len(), src);
    }
    decompress_zip(dst, src) == Some(dst.len())
}

/// `DecodeDeepTiled`: a single-level deep *tiled* part into the scanline-organized
/// `DeepImage` (`offset_table`[y][x] = cumulative count within row y; image[c][y] = that row's
/// samples concatenated in x order). `marker` is where the tile offset table is.
// (Its parameters are tinyexr's, each a separate part of the state; one pass, step for step as
// tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_deep_tiled(
    memory: &Memory,
    channels: &[ChannelInfo],
    num_channels: i32,
    compression_type: Compression,
    data_width: i32,
    data_height: i32,
    tile_size_x: i32,
    tile_size_y: i32,
    tile_level_mode: TileLevelMode,
    mut marker: i64,
) -> Result<DeepImage, Stop> {
    // A tile's place, its part of the image, and its block's sizes, as the original reads
    // them.
    struct Tile {
        offset: u64,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
        pots: i64,
        psds: i64,
        usds: i64,
    }
    // A tile's place and level, and the sizes of its offset table and of its samples (packed
    // and unpacked).
    type TileHeader = (i32, i32, i32, i32, i64, i64, i64);
    let file_size = memory.bytes.len() as u64;
    if tile_size_x <= 0 || tile_size_y <= 0 {
        return Err(error(
            ErrorCode::INVALID_DATA,
            "Invalid tile size in deep tiled image",
        ));
    }
    if tile_level_mode != TileLevelMode(0) {
        return Err(error(
            ErrorCode::UNSUPPORTED_FORMAT,
            "Only single-level (ONE_LEVEL) deep tiled images are supported",
        ));
    }
    if data_width <= 0 || data_height <= 0 || num_channels < 1 {
        return Err(error(
            ErrorCode::INVALID_DATA,
            "Invalid deep tiled image dimensions",
        ));
    }
    if !(compression_type == Compression::NONE
        || compression_type == Compression::RLE
        || compression_type == Compression::ZIPS
        || compression_type == Compression::ZIP)
    {
        return Err(error(
            ErrorCode::UNSUPPORTED_FORMAT,
            "Unsupported compression for deep tiled image",
        ));
    }

    let mut ch_size = Vec::with_capacity(channels.len());
    let mut sample_size = 0i32;
    for channel in channels {
        let ps = match channel.pixel_type {
            PixelType::HALF => 2,
            PixelType::UINT | PixelType::FLOAT => 4,
            _ => {
                return Err(error(
                    ErrorCode::INVALID_DATA,
                    "Invalid pixel type in deep tiled image",
                ));
            }
        };
        ch_size.push(ps);
        sample_size += ps;
    }

    let num_x_tiles = (i64::from(data_width) + i64::from(tile_size_x) - 1) / i64::from(tile_size_x);
    let num_y_tiles =
        (i64::from(data_height) + i64::from(tile_size_y) - 1) / i64::from(tile_size_y);
    let num_tiles = num_x_tiles as u64 * num_y_tiles as u64;
    let (num_x_tiles, num_y_tiles) = (num_x_tiles as i32, num_y_tiles as i32);

    if marker as u64 + num_tiles.saturating_mul(8) > file_size {
        return Err(error(
            ErrorCode::INVALID_DATA,
            "Corrupt deep tiled offset table",
        ));
    }
    let mut tile_offsets = Vec::with_capacity(num_tiles as usize);
    for _ in 0..num_tiles {
        tile_offsets.push(memory.i64_at(marker)? as u64);
        marker += 8;
    }

    let (width, height) = (data_width as usize, data_height as usize);
    if width as u64 * height as u64 > MAX_PIXELS {
        return Err(Stop::Outside);
    }
    let mut counts = vec![0i32; width * height];

    let tile_at = |offset: u64| -> Result<TileHeader, Stop> {
        let p = offset as i64;
        Ok((
            memory.i32_at(p)?,
            memory.i32_at(p + 4)?,
            memory.i32_at(p + 8)?,
            memory.i32_at(p + 12)?,
            memory.i64_at(p + 16)?,
            memory.i64_at(p + 24)?,
            memory.i64_at(p + 32)?,
        ))
    };
    let place = |tile_x: i32, tile_y: i32| {
        let (x0, y0) = (tile_x * tile_size_x, tile_y * tile_size_y);
        let tw = tile_size_x.min(data_width - x0);
        let th = tile_size_y.min(data_height - y0);
        (x0 as usize, y0 as usize, tw as usize, th as usize)
    };

    // ---- Pass 1: per-tile offset tables -> per-pixel sample counts ----
    let mut tiles = Vec::with_capacity(tile_offsets.len());
    for &offset in &tile_offsets {
        if offset.checked_add(40).is_none_or(|end| end > file_size) {
            return Err(error(ErrorCode::INVALID_DATA, "Corrupt deep tile chunk"));
        }
        let (tile_x, tile_y, level_x, level_y, pots, psds, usds) = tile_at(offset)?;
        if level_x != 0 || level_y != 0 {
            return Err(error(
                ErrorCode::UNSUPPORTED_FORMAT,
                "Unsupported deep tile level",
            ));
        }
        if tile_x < 0 || tile_y < 0 || tile_x >= num_x_tiles || tile_y >= num_y_tiles || pots < 0 {
            return Err(error(ErrorCode::INVALID_DATA, "Corrupt deep tile header"));
        }
        let otab_packed = offset + 40;
        if otab_packed + pots as u64 > file_size {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Corrupt deep tile offset table",
            ));
        }
        let (x0, y0, tw, th) = place(tile_x, tile_y);
        let need = tw * th;
        let mut otab = vec![0u8; need * 4];
        let src = memory.read(otab_packed as i64, pots as u64)?;
        if !decompress_deep(compression_type, &mut otab, &src) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Failed to decompress deep tile offset table",
            ));
        }
        // The sample-count table is cumulative per scanline row within the tile.
        for ry in 0..th {
            let mut prev = 0i32;
            for rx in 0..tw {
                let at = (ry * tw + rx) * 4;
                let cum = i32::from_le_bytes(otab[at..at + 4].try_into().unwrap());
                let cnt = cum.wrapping_sub(prev);
                if cnt < 0 {
                    return Err(error(
                        ErrorCode::INVALID_DATA,
                        "Corrupt deep tile (negative sample count)",
                    ));
                }
                counts[(y0 + ry) * width + x0 + rx] = cnt;
                prev = cum;
            }
        }
        tiles.push(Tile {
            offset,
            x0,
            y0,
            tw,
            th,
            pots,
            psds,
            usds,
        });
    }

    // ---- Build offset_table[y][x] ----
    let mut deep_image = DeepImage::default();
    let mut row_samples = vec![0usize; height];
    for y in 0..height {
        let mut row = Vec::with_capacity(width);
        let mut run = 0i32;
        for x in 0..width {
            run = run.wrapping_add(counts[y * width + x]);
            row.push(run);
        }
        row_samples[y] = run as isize as usize;
        deep_image.offset_table.push(row);
    }

    // ---- Pass 2: per-tile (channel-planar) sample data ----
    // (Every block is checked and decompressed before the rows are made: the original makes
    // them first, but nothing it does with them can fail, and a block that does not add up is
    // found before rows of more samples than there can be are made.)
    let mut blocks = Vec::with_capacity(tiles.len());
    for tile in &tiles {
        let Tile {
            offset,
            x0,
            y0,
            tw,
            th,
            pots,
            psds,
            usds,
        } = *tile;
        if psds < 0 || usds < 0 {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Corrupt deep tile sample sizes",
            ));
        }
        let samp_packed = offset + 40 + pots as u64;
        if samp_packed + psds as u64 > file_size {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Corrupt deep tile sample data",
            ));
        }

        let mut tile_total = 0u64;
        for ry in 0..th {
            for rx in 0..tw {
                tile_total =
                    tile_total.wrapping_add(i64::from(counts[(y0 + ry) * width + x0 + rx]) as u64);
            }
        }
        if usds != (tile_total as i64).wrapping_mul(i64::from(sample_size)) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Deep tile sample size mismatch",
            ));
        }
        let src = memory.read(samp_packed as i64, psds as u64)?;
        let usds = usds as u64;
        // (A block that says more than its bytes can come to does not come out that long.)
        if usds != src.len() as u64 && usds > most_from(src.len()) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Failed to decompress deep tile sample data",
            ));
        }
        let mut sbuf = vec![0u8; usds as usize];
        if !decompress_deep(compression_type, &mut sbuf, &src) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Failed to decompress deep tile sample data",
            ));
        }
        blocks.push(sbuf);
    }

    // ---- Allocate per-row sample arrays and scatter ----
    deep_image.image = (0..num_channels)
        .map(|_| row_samples.iter().map(|&n| vec![0f32; n]).collect())
        .collect();
    for (tile, sbuf) in tiles.iter().zip(&blocks) {
        let Tile { x0, y0, tw, th, .. } = *tile;
        // Sample data is stored one tile-row at a time, each row channel-planar
        // (i.e. the tile is a stack of `th` deep "scanlines" of width `tw`), which
        // is why the offset table is cumulative per row.
        let mut offset = 0usize; // byte offset within sbuf
        for ry in 0..th {
            let gy = y0 + ry;
            for (c, channel) in channels.iter().enumerate() {
                let ps = ch_size[c] as usize;
                let mut local = 0usize; // sample index within this (row, channel) block
                for rx in 0..tw {
                    let gx = x0 + rx;
                    let cnt = counts[gy * width + gx] as usize;
                    let row_start =
                        (deep_image.offset_table[gy][gx] as isize - cnt as isize) as usize;
                    let sp = offset + local * ps;
                    let dst = &mut deep_image.image[c][gy][row_start..row_start + cnt];
                    for (s, value) in dst.iter_mut().enumerate() {
                        let at = sp + s * ps;
                        *value = match channel.pixel_type {
                            PixelType::UINT => {
                                u32::from_le_bytes(sbuf[at..at + 4].try_into().unwrap()) as f32
                            }
                            PixelType::HALF => half_to_float(u16::from_le_bytes(
                                sbuf[at..at + 2].try_into().unwrap(),
                            )),
                            _ => f32::from_le_bytes(sbuf[at..at + 4].try_into().unwrap()),
                        };
                    }
                    local += cnt;
                }
                offset += local * ps;
            }
        }
    }

    deep_image.width = data_width;
    deep_image.height = data_height;
    deep_image.num_channels = num_channels;
    deep_image.channel_names = channels
        .iter()
        .map(|channel| channel.name.clone())
        .collect();
    Ok(deep_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_nrp as miniz;

    /// A deep file of scanlines, ZIPS: two lines of two pixels, one float channel `Z`.
    fn deep_file() -> Vec<u8> {
        let attribute = |out: &mut Vec<u8>, name: &str, kind: &str, data: &[u8]| {
            out.extend_from_slice(name.as_bytes());
            out.push(0);
            out.extend_from_slice(kind.as_bytes());
            out.push(0);
            out.extend_from_slice(&(data.len() as i32).to_le_bytes());
            out.extend_from_slice(data);
        };
        let mut file = vec![0x76, 0x2f, 0x31, 0x01, 2, 8, 0, 0];
        let mut chlist = b"Z\0".to_vec();
        chlist.extend_from_slice(&2i32.to_le_bytes());
        chlist.extend_from_slice(&[0; 4]);
        chlist.extend_from_slice(&1i32.to_le_bytes());
        chlist.extend_from_slice(&1i32.to_le_bytes());
        chlist.push(0);
        attribute(&mut file, "channels", "chlist", &chlist);
        attribute(
            &mut file,
            "compression",
            "compression",
            &[Compression::ZIPS.0 as u8],
        );
        let window: Vec<u8> = [0i32, 0, 1, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        attribute(&mut file, "dataWindow", "box2i", &window);
        attribute(&mut file, "displayWindow", "box2i", &window);
        file.push(0);
        // Line y has pixel counts 1 and y + 1: cumulative 1, y + 2.
        let table_at = file.len();
        file.extend_from_slice(&[0; 16]);
        for y in 0..2u8 {
            let at = file.len() as u64;
            file[table_at + 8 * y as usize..table_at + 8 * y as usize + 8]
                .copy_from_slice(&at.to_le_bytes());
            let cumulative: Vec<u8> = [1i32, i32::from(y) + 2]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let samples: Vec<u8> = (0..i32::from(y) + 2)
                .flat_map(|s| (s as f32 + 10.0 * f32::from(y)).to_le_bytes())
                .collect();
            // (Stored as they are: a block as long as its bytes is taken for them.)
            let packed_table = cumulative.clone();
            let packed_samples = if y == 0 {
                samples.clone()
            } else {
                miniz::compress(&predict(&samples), miniz::Level::new(6).unwrap())
            };
            file.extend_from_slice(&i32::from(y).to_le_bytes());
            file.extend_from_slice(&(packed_table.len() as i64).to_le_bytes());
            file.extend_from_slice(&(packed_samples.len() as i64).to_le_bytes());
            file.extend_from_slice(&(samples.len() as i64).to_le_bytes());
            file.extend_from_slice(&packed_table);
            file.extend_from_slice(&packed_samples);
        }
        file
    }

    /// ZIP's reordering and predictor, as a writer applies them.
    fn predict(src: &[u8]) -> Vec<u8> {
        let half = src.len().div_ceil(2);
        let mut t = vec![0u8; src.len()];
        for (i, &byte) in src.iter().enumerate() {
            t[if i % 2 == 0 { i / 2 } else { half + i / 2 }] = byte;
        }
        let mut p = t[0];
        for byte in t.iter_mut().skip(1) {
            let d = byte.wrapping_sub(p).wrapping_add(128);
            p = *byte;
            *byte = d;
        }
        t
    }

    #[test]
    fn a_deep_file_of_scanlines_is_read() {
        let image = load_deep(&deep_file()).unwrap();
        assert_eq!(
            (
                image.width,
                image.height,
                image.num_channels,
                image.channel_names.clone()
            ),
            (2, 2, 1, vec!["Z".to_string()])
        );
        assert_eq!(image.offset_table, [[1, 2], [1, 3]]);
        assert_eq!(image.image[0], [vec![0.0, 1.0], vec![10.0, 11.0, 12.0]]);
    }

    #[test]
    fn a_deep_file_cut_short_is_refused() {
        let file = deep_file();
        for len in [0, 3, 7, 40, file.len() - 30, file.len() - 1] {
            // (Refused, or, where a compressed block is cut, handed back unfinished, as the
            // original does: with no width.)
            assert!(
                load_deep(&file[..len]).is_ok_and(|image| image.width == 0)
                    || load_deep(&file[..len]).is_err(),
                "{len}"
            );
        }
    }
}
