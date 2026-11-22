//! Deep images: a port of `LoadDeepEXR` and `DecodeDeepTiled` of tinyexr v3.4.0
//! (`tinyexr.h`), read from memory, plus [`save_deep`] for writing deep scanline `.exr` files.
//!
//! What tinyexr reads, and so the port:
//!
//! - **Deep scanlines** whose blocks are ZIPS-compressed (one line a block). The reader takes
//!   each block for one line, so with ZIP (sixteen lines a block) only the first of every
//!   sixteen lines has samples, and the others none. A block it cannot decompress ends the
//!   reading there and is not reported: the image comes back empty, with no width, height,
//!   channels or samples.
//! - **Deep tiles** of one level, uncompressed or compressed with RLE, ZIPS or ZIP.

use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cell::Cell;

use super::parts::PAST_THE_END;
use super::zip_rle::{compress_zip_into, decompress_rle, decompress_zip, unpredict_and_reorder};
use super::{
    ChannelInfo, Compression, ErrorCode, ExrError, PixelType, TileLevelMode, half_to_float,
    read_attribute, read_channel_info,
};
use miniz_nrp::{DecompressError, Format};

/// `DeepImage`: the samples of a deep image, line by line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeepImage {
    /// The name of each channel.
    pub channel_names: Vec<String>,
    /// `image[channel][line]`: the samples of a line, every pixel's one after another.
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
/// and on to the next multiple of 4 KiB from the start of the file.
fn reach(size: u64) -> u64 {
    (size + 65536).div_ceil(4096) * 4096
}

/// The largest number of pixels a deep image is read or written with: 67 million.
const MAX_PIXELS: u64 = 1 << 26;

/// The most bytes miniz's `mz_uncompress` takes in: past it, it fails at once.
const MZ_LIMIT: u64 = 0xFFFF_FFFF;

/// The largest block of a file of scanlines (or total deep tile sample memory) that is
/// decompressed: 1 GiB.
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

/// Writes `deep_image` as an OpenEXR 2.0 `deepscanline` (ZIPS-compressed, `FLOAT` channels)
/// `.exr` file in memory that round-trips through [`load_deep`].
///
/// # Errors
///
/// Returns [`ErrorCode::INVALID_ARGUMENT`] if dimensions, channel names, cumulative
/// `offset_table` rows, or `image` sample counts are missing or inconsistent, or
/// [`ErrorCode::DATA_TOO_LARGE`] if the image exceeds [`load_deep`]'s size bounds.
#[allow(clippy::too_many_lines)]
pub fn save_deep(deep_image: &DeepImage) -> Result<Vec<u8>, ExrError> {
    let bad = |message: &str| {
        Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: message.to_string(),
        })
    };
    if deep_image.width < 1 || deep_image.height < 1 {
        return bad("A deep image needs a positive width and height.");
    }
    if deep_image.num_channels < 1 {
        return bad("A deep image needs at least one channel.");
    }
    let width = deep_image.width as usize;
    let height = deep_image.height as usize;
    let num_channels = deep_image.num_channels as usize;
    if (width as u64).saturating_mul(height as u64) > MAX_PIXELS {
        return Err(ExrError::new(
            ErrorCode::DATA_TOO_LARGE,
            "Deep image pixel count exceeds maximum.",
        ));
    }
    if deep_image.channel_names.len() != num_channels || deep_image.image.len() != num_channels {
        return bad("channel_names and image must match num_channels.");
    }
    if deep_image.offset_table.len() != height {
        return bad("offset_table must have one row per scanline.");
    }

    let mut clean_names: Vec<Vec<u8>> = Vec::with_capacity(num_channels);
    for raw_name in &deep_image.channel_names {
        let bytes = raw_name.as_bytes();
        let end = bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(bytes.len())
            .min(255);
        let slice = &bytes[..end];
        if slice.is_empty() {
            return bad("Deep channel names must not be empty.");
        }
        if clean_names.iter().any(|seen| seen.as_slice() == slice) {
            return bad("Deep channel names must be unique.");
        }
        clean_names.push(slice.to_vec());
    }

    let mut max_samples_per_pixel = 0i32;
    let mut line_totals = Vec::with_capacity(height);
    for (y, row) in deep_image.offset_table.iter().enumerate() {
        if row.len() != width {
            return bad("Each offset_table row must have `width` cumulative sample counts.");
        }
        let mut prev = 0i32;
        for &cum in row {
            if cum < prev {
                return bad("offset_table cumulative sample counts must be non-decreasing.");
            }
            let cnt = cum - prev;
            if cnt > max_samples_per_pixel {
                max_samples_per_pixel = cnt;
            }
            prev = cum;
        }
        let total_on_line = prev as usize;
        let unpacked_bytes = (total_on_line as u64)
            .saturating_mul(num_channels as u64)
            .saturating_mul(4);
        if unpacked_bytes > MAX_BLOCK {
            return Err(ExrError::new(
                ErrorCode::DATA_TOO_LARGE,
                "Deep scanline sample block exceeds 1 GiB.",
            ));
        }
        for ch_rows in &deep_image.image {
            if ch_rows.len() != height || ch_rows[y].len() != total_on_line {
                return bad(
                    "Each channel's scanline sample vector must match offset_table's line total.",
                );
            }
        }
        line_totals.push(total_on_line);
    }

    let write_attr = |out: &mut Vec<u8>, name: &[u8], kind: &[u8], data: &[u8]| {
        out.extend_from_slice(name);
        out.push(0);
        out.extend_from_slice(kind);
        out.push(0);
        out.extend_from_slice(&(data.len() as i32).to_le_bytes());
        out.extend_from_slice(data);
    };

    // Magic + EXR 2.0 single-part deep scanline version (`[2, 8, 0, 0]`).
    let mut out = vec![0x76, 0x2f, 0x31, 0x01, 2, 8, 0, 0];

    let mut chlist = Vec::new();
    for name in &clean_names {
        chlist.extend_from_slice(name);
        chlist.push(0);
        chlist.extend_from_slice(&PixelType::FLOAT.0.to_le_bytes());
        chlist.extend_from_slice(&[0, 0, 0, 0]);
        chlist.extend_from_slice(&1i32.to_le_bytes());
        chlist.extend_from_slice(&1i32.to_le_bytes());
    }
    chlist.push(0);
    write_attr(&mut out, b"channels", b"chlist", &chlist);
    write_attr(
        &mut out,
        b"compression",
        b"compression",
        &[Compression::ZIPS.0 as u8],
    );

    let mut window = Vec::with_capacity(16);
    for v in [0i32, 0, deep_image.width - 1, deep_image.height - 1] {
        window.extend_from_slice(&v.to_le_bytes());
    }
    write_attr(&mut out, b"dataWindow", b"box2i", &window);
    write_attr(&mut out, b"displayWindow", b"box2i", &window);
    write_attr(&mut out, b"lineOrder", b"lineOrder", &[0]);
    write_attr(
        &mut out,
        b"pixelAspectRatio",
        b"float",
        &1.0f32.to_le_bytes(),
    );
    write_attr(&mut out, b"screenWindowCenter", b"v2f", &[0u8; 8]);
    write_attr(
        &mut out,
        b"screenWindowWidth",
        b"float",
        &1.0f32.to_le_bytes(),
    );
    write_attr(&mut out, b"name", b"string", b"deep");
    write_attr(&mut out, b"type", b"string", b"deepscanline");
    write_attr(&mut out, b"version", b"int", &1i32.to_le_bytes());
    write_attr(
        &mut out,
        b"chunkCount",
        b"int",
        &deep_image.height.to_le_bytes(),
    );
    write_attr(
        &mut out,
        b"maxSamplesPerPixel",
        b"int",
        &max_samples_per_pixel.to_le_bytes(),
    );
    out.push(0); // end of header

    let offset_table_pos = out.len();
    out.resize(offset_table_pos + height * 8, 0);

    let mut compressor = miniz_nrp::Compressor::default();
    let mut tmp_buf = Vec::new();
    let mut raw_otab = vec![0u8; width * 4];
    let mut packed_otab = Vec::new();
    let mut raw_samples = Vec::new();
    let mut packed_samples = Vec::new();

    for (y, &samples_on_line) in line_totals.iter().enumerate() {
        let chunk_offset = out.len() as u64;
        out[offset_table_pos + y * 8..offset_table_pos + y * 8 + 8]
            .copy_from_slice(&chunk_offset.to_le_bytes());

        for (slot, &cum) in raw_otab
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&deep_image.offset_table[y])
        {
            *slot = cum.to_le_bytes();
        }
        packed_otab.clear();
        compress_zip_into(&raw_otab, &mut tmp_buf, &mut compressor, &mut packed_otab);

        let unpacked_sample_bytes = samples_on_line * num_channels * 4;
        packed_samples.clear();
        if unpacked_sample_bytes > 0 {
            raw_samples.clear();
            raw_samples.resize(unpacked_sample_bytes, 0);
            let mut cursor = 0usize;
            for ch_rows in &deep_image.image {
                let src = &ch_rows[y];
                for (slot, &val) in raw_samples[cursor..cursor + samples_on_line * 4]
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(src)
                {
                    *slot = val.to_bits().to_le_bytes();
                }
                cursor += samples_on_line * 4;
            }
            compress_zip_into(
                &raw_samples,
                &mut tmp_buf,
                &mut compressor,
                &mut packed_samples,
            );
        }

        out.extend_from_slice(&(y as i32).to_le_bytes());
        out.extend_from_slice(&(packed_otab.len() as i64).to_le_bytes());
        out.extend_from_slice(&(packed_samples.len() as i64).to_le_bytes());
        out.extend_from_slice(&(unpacked_sample_bytes as i64).to_le_bytes());
        out.extend_from_slice(&packed_otab);
        out.extend_from_slice(&packed_samples);
    }

    Ok(out)
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
#[allow(clippy::too_many_lines)]
fn load(memory: &Memory) -> Result<DeepImage, Stop> {
    let file_size = memory.bytes.len();
    if file_size == 0 {
        return Err(error(ErrorCode::INVALID_FILE, "File size is zero"));
    }

    if memory.read(0, 4)?[..] != [0x76, 0x2f, 0x31, 0x01] {
        return Err(error(
            ErrorCode::INVALID_MAGIC_NUMBER,
            "Invalid magic number",
        ));
    }
    let mut marker = 4i64;

    {
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
    let mut num_scanline_blocks = 1i32;
    let mut compression_type = Compression(-1);
    let mut num_channels = -1i32;
    let mut is_tiled = false;
    let mut tile_size_x = -1i32;
    let mut tile_size_y = -1i32;
    let mut tile_level_mode = 0i32;
    let mut channels: Vec<ChannelInfo> = Vec::new();

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
                need(13)?;
            }
            "tiles" => {
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

    let mut num_blocks = data_height / num_scanline_blocks;
    if num_blocks.wrapping_mul(num_scanline_blocks) < data_height {
        num_blocks += 1;
    }
    if num_blocks < 0 {
        return Err(Stop::Outside);
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
        return Err(Stop::Outside);
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

        let packed_offset_table_size = memory.i64_at(offset.wrapping_add(4))?;
        let packed_sample_data_size = memory.i64_at(offset.wrapping_add(12))?;
        let unpacked_sample_data_size = memory.i64_at(offset.wrapping_add(20))?;

        let mut pixel_offset_table = vec![0i32; width];

        {
            let Some(table) = decompress_zip_from(
                memory,
                offset.wrapping_add(28),
                packed_offset_table_size as u64,
                width as u64 * 4,
            )?
            else {
                return Ok(unfinished());
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

        if unpacked_sample_data_size < 0 || unpacked_sample_data_size as u64 > MAX_BLOCK {
            return Err(Stop::Outside);
        }
        let mut sample_data = Vec::new();

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
        let samples_per_line = sample_data.len() / (sample_size as usize);

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
                    .collect(),
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
    }

    deep_image.width = data_width;
    deep_image.height = data_height;
    deep_image.channel_names = channels
        .iter()
        .map(|channel| channel.name.clone())
        .collect();
    deep_image.num_channels = num_channels;

    Ok(deep_image)
}

fn unfinished() -> DeepImage {
    DeepImage::default()
}

fn most_from(src_len: usize) -> u64 {
    (src_len as u64).saturating_mul(1032).saturating_add(1024)
}

fn decompress_zip_from(
    memory: &Memory,
    at: i64,
    src_len: u64,
    dst_len: u64,
) -> Result<Option<Vec<u8>>, Stop> {
    if dst_len == src_len {
        return Ok(Some(memory.read(at, src_len)?.into_owned()));
    }
    if dst_len == 0 || dst_len > MAX_BLOCK {
        return Err(Stop::Outside);
    }
    let Ok(dst_usize) = usize::try_from(dst_len) else {
        return Err(Stop::Outside);
    };
    if src_len > MZ_LIMIT {
        return Ok(None);
    }
    let src = memory.read_up_to(at, src_len)?;
    let room = dst_usize.min(usize::try_from(most_from(src.len())).unwrap_or(usize::MAX));
    match miniz_nrp::decompress_with(&src, Format::Zlib, room) {
        Ok(result) => {
            let mut tmp_buf = result.data;
            let mut dst = vec![0u8; tmp_buf.len()];
            unpredict_and_reorder(&mut dst, &mut tmp_buf);
            Ok(Some(dst))
        }
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

/// `DecodeDeepTiled`: a single-level deep *tiled* part into the scanline-organized `DeepImage`.
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
        sample_size = sample_size
            .checked_add(ps)
            .ok_or_else(|| error(ErrorCode::INVALID_DATA, "Too many channels in deep tile"))?;
    }

    let num_x_tiles = (i64::from(data_width) + i64::from(tile_size_x) - 1) / i64::from(tile_size_x);
    let num_y_tiles =
        (i64::from(data_height) + i64::from(tile_size_y) - 1) / i64::from(tile_size_y);
    let num_tiles = num_x_tiles as u64 * num_y_tiles as u64;
    let (num_x_tiles, num_y_tiles) = (num_x_tiles as i32, num_y_tiles as i32);

    if (marker as u64)
        .checked_add(num_tiles.saturating_mul(8))
        .is_none_or(|end| end > file_size)
    {
        return Err(error(
            ErrorCode::INVALID_DATA,
            "Corrupt deep tiled offset table",
        ));
    }
    let Ok(num_tiles_usize) = usize::try_from(num_tiles) else {
        return Err(Stop::Outside);
    };
    let mut tile_offsets = Vec::with_capacity(num_tiles_usize.min(1 << 20));
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
        let x0 = (i64::from(tile_x) * i64::from(tile_size_x)) as usize;
        let y0 = (i64::from(tile_y) * i64::from(tile_size_y)) as usize;
        let tw = (tile_size_x as usize).min(width - x0);
        let th = (tile_size_y as usize).min(height - y0);
        (x0, y0, tw, th)
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
        if otab_packed
            .checked_add(pots as u64)
            .is_none_or(|end| end > file_size)
        {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Corrupt deep tile offset table",
            ));
        }
        let (x0, y0, tw, th) = place(tile_x, tile_y);
        let Some(otab_bytes) = tw.checked_mul(th).and_then(|n| n.checked_mul(4)) else {
            return Err(Stop::Outside);
        };
        let mut otab = vec![0u8; otab_bytes];
        let src = memory.read(otab_packed as i64, pots as u64)?;
        if !decompress_deep(compression_type, &mut otab, &src) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Failed to decompress deep tile offset table",
            ));
        }
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

    // ---- Build offset_table[y][x] with checked arithmetic ----
    let mut deep_image = DeepImage::default();
    let mut row_samples = vec![0usize; height];
    let mut total_sample_bytes = 0u64;
    for y in 0..height {
        let mut row = Vec::with_capacity(width);
        let mut run = 0i32;
        for x in 0..width {
            let Some(next_run) = run.checked_add(counts[y * width + x]) else {
                return Err(error(
                    ErrorCode::INVALID_DATA,
                    "Corrupt deep tile (sample count overflow)",
                ));
            };
            run = next_run;
            row.push(run);
        }
        let n = run as usize;
        row_samples[y] = n;
        total_sample_bytes = total_sample_bytes.saturating_add(
            (n as u64)
                .saturating_mul(num_channels as u64)
                .saturating_mul(4),
        );
        if total_sample_bytes > MAX_BLOCK {
            return Err(Stop::Outside);
        }
        deep_image.offset_table.push(row);
    }

    // ---- Pass 2: per-tile (channel-planar) sample data ----
    let mut blocks = Vec::with_capacity(tiles.len());
    let mut total_block_bytes = 0u64;
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
        let Some(samp_packed) = offset
            .checked_add(40)
            .and_then(|v| v.checked_add(pots as u64))
        else {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Corrupt deep tile sample data",
            ));
        };
        if samp_packed
            .checked_add(psds as u64)
            .is_none_or(|end| end > file_size)
        {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Corrupt deep tile sample data",
            ));
        }

        let mut tile_total = 0u64;
        for ry in 0..th {
            for rx in 0..tw {
                tile_total = tile_total.saturating_add(counts[(y0 + ry) * width + x0 + rx] as u64);
            }
        }
        let expected_usds = tile_total.checked_mul(sample_size as u64);
        if expected_usds != Some(usds as u64) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Deep tile sample size mismatch",
            ));
        }
        let usds_u64 = usds as u64;
        if usds_u64 > MAX_BLOCK {
            return Err(Stop::Outside);
        }
        total_block_bytes = total_block_bytes.saturating_add(usds_u64);
        if total_block_bytes > MAX_BLOCK {
            return Err(Stop::Outside);
        }
        let Ok(usds_usize) = usize::try_from(usds_u64) else {
            return Err(Stop::Outside);
        };

        let src = memory.read(samp_packed as i64, psds as u64)?;
        if usds_u64 != src.len() as u64 && usds_u64 > most_from(src.len()) {
            return Err(error(
                ErrorCode::INVALID_DATA,
                "Failed to decompress deep tile sample data",
            ));
        }
        let mut sbuf = vec![0u8; usds_usize];
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
        let mut offset = 0usize;
        for ry in 0..th {
            let gy = y0 + ry;
            for (c, channel) in channels.iter().enumerate() {
                let ps = ch_size[c] as usize;
                let mut local = 0usize;
                for rx in 0..tw {
                    let gx = x0 + rx;
                    let cnt = counts[gy * width + gx] as usize;
                    let cum = deep_image.offset_table[gy][gx] as usize;
                    let Some(row_start) = cum.checked_sub(cnt) else {
                        return Err(error(ErrorCode::INVALID_DATA, "Corrupt deep tile offsets"));
                    };
                    let Some(sp) = local.checked_mul(ps).and_then(|v| offset.checked_add(v)) else {
                        return Err(error(ErrorCode::INVALID_DATA, "Corrupt deep tile offsets"));
                    };
                    let Some(dst) = deep_image.image[c][gy].get_mut(row_start..row_start + cnt)
                    else {
                        return Err(error(ErrorCode::INVALID_DATA, "Corrupt deep tile offsets"));
                    };
                    for (s, value) in dst.iter_mut().enumerate() {
                        let at = sp + s * ps;
                        let Some(bytes) = sbuf.get(at..at + ps) else {
                            return Err(error(
                                ErrorCode::INVALID_DATA,
                                "Corrupt deep tile sample data",
                            ));
                        };
                        *value = match channel.pixel_type {
                            PixelType::UINT => u32::from_le_bytes(bytes.try_into().unwrap()) as f32,
                            PixelType::HALF => {
                                half_to_float(u16::from_le_bytes(bytes.try_into().unwrap()))
                            }
                            _ => f32::from_le_bytes(bytes.try_into().unwrap()),
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
    fn save_deep_round_trips_through_load_deep_and_validates_inputs() {
        let orig = DeepImage {
            channel_names: vec!["A".to_string(), "Z".to_string()],
            image: vec![
                vec![vec![0.25, 0.75, 1.0], vec![], vec![0.5]],
                vec![vec![1.5, 2.5, 9.0], vec![], vec![-0.0]],
            ],
            offset_table: vec![vec![1, 1, 3, 3], vec![0, 0, 0, 0], vec![0, 1, 1, 1]],
            num_channels: 2,
            width: 4,
            height: 3,
        };
        let encoded = save_deep(&orig).unwrap();
        let ver = crate::parse_version(&encoded).unwrap();
        assert!(ver.non_image && !ver.tiled && !ver.multipart);
        let hdr = crate::parse_header(ver, &encoded).unwrap();
        assert!(hdr.non_image);
        assert_eq!(hdr.channels.len(), 2);

        let decoded = load_deep(&encoded).unwrap();
        assert!(same(&orig, &decoded));

        // Invalid inputs are cleanly rejected.
        let mut bad = orig.clone();
        bad.offset_table[0] = vec![2, 1, 3, 3]; // decreasing cumulative count
        assert_eq!(
            save_deep(&bad).unwrap_err().code,
            ErrorCode::INVALID_ARGUMENT
        );

        let mut dup = orig.clone();
        dup.channel_names = vec!["Z".to_string(), "Z".to_string()];
        assert_eq!(
            save_deep(&dup).unwrap_err().code,
            ErrorCode::INVALID_ARGUMENT
        );
    }

    #[test]
    fn a_deep_file_cut_short_is_refused() {
        let file = deep_file();
        for len in [0, 3, 7, 40, file.len() - 30, file.len() - 1] {
            assert!(
                load_deep(&file[..len]).is_ok_and(|image| image.width == 0)
                    || load_deep(&file[..len]).is_err(),
                "{len}"
            );
        }
    }
}
