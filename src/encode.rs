//! Writing a file: a port of `SaveEXRImageToMemory` and `SaveEXRMultipartImageToMemory`
//! (`SaveEXRNPartImageToMemory`), `EncodeChunk`, `EncodeTiledLevel`, `EncodePixelData` and
//! `SaveEXRToMemory` of tinyexr v3.4.0 (`tinyexr.h`).

use super::b44::compress_b44;
use super::chunk::{
    OffsetData, level_index, num_scanline_blocks, precalculate_tile_info, tile_offsets_shape,
};
use super::decode::ChannelData;
use super::piz::compress_piz;
use super::pxr24::compress_pxr24;
use super::zfp::{ZfpParam, compress_zfp, zfp_param_for_writing, zfp_stream_size};
use super::zip_rle::{compress_rle, compress_zip};
use super::{
    Attribute, ChannelInfo, Compression, ErrorCode, ExrError, ExrHeader, ExrImage, ExrTile,
    PixelType, TileLevelMode, float_to_half, half_to_float,
};
use crate::TileRoundingMode;
use openexr_nrp::{PixelType as DwaPixelType, dwa};

/// `WriteAttributeToMemory`.
fn write_attribute_to_memory(out: &mut Vec<u8>, name: &str, type_name: &str, data: &[u8]) {
    // (The original's names are C strings: they end at a zero.)
    let c_string = |text: &str| {
        text.as_bytes()
            .iter()
            .copied()
            .take_while(|&byte| byte != 0)
            .collect::<Vec<u8>>()
    };
    out.extend_from_slice(&c_string(name));
    out.push(0);
    out.extend_from_slice(&c_string(type_name));
    out.push(0);

    out.extend_from_slice(&(data.len() as i32).to_le_bytes());
    out.extend_from_slice(data);
}

/// `WriteAttributeToMemory` of a name that is bytes.
fn write_attribute_bytes(out: &mut Vec<u8>, name: &[u8], type_name: &[u8], data: &[u8]) {
    out.extend_from_slice(name);
    out.push(0);
    out.extend_from_slice(type_name);
    out.push(0);
    out.extend_from_slice(&(data.len() as i32).to_le_bytes());
    out.extend_from_slice(data);
}

/// `WriteChannelInfo`: the list of channels as a file has it, each with the type it is
/// stored as.
fn write_channel_info(channels: &[ChannelInfo], file_types: &[PixelType]) -> Vec<u8> {
    let mut data = Vec::new();
    for (channel, &file_type) in channels.iter().zip(file_types) {
        data.extend(
            channel
                .name
                .as_bytes()
                .iter()
                .copied()
                .take_while(|&byte| byte != 0),
        );
        data.push(0);

        data.extend_from_slice(&file_type.0.to_le_bytes());
        // (The writer's channels are linear, and sampled at every pixel.)
        data.extend_from_slice(&[0, 0, 0, 0]);
        data.extend_from_slice(&1i32.to_le_bytes());
        data.extend_from_slice(&1i32.to_le_bytes());
    }
    data.push(0);
    data
}

/// `EncodePixelData`: `num_lines` lines of `width` pixels from the line `line_no` on, taken
/// from `images` (whose lines are `x_stride` values apart), as a block of a file, added to
/// `out_data`. Each channel is stored as its type in `file_types`.
#[allow(clippy::too_many_arguments)]
fn encode_pixel_data(
    out_data: &mut Vec<u8>,
    images: &[ChannelData],
    compression_type: Compression,
    width: i32,
    x_stride: i32,
    line_no: i32,
    num_lines: i32,
    pixel_data_size: usize,
    file_types: &[PixelType],
    channel_offset_list: &[usize],
    zfp: &ZfpParam,
    dwa: &DwaParam,
    err: &mut String,
    as_tinyexr: bool,
) -> bool {
    let width_size = width as usize;
    let buf_size = width_size * num_lines as usize * pixel_data_size;
    let mut buf = vec![0u8; buf_size];

    let start_y = line_no as usize;
    for (c, image) in images.iter().enumerate() {
        let file_type = file_types[c];
        if !matches!(image, ChannelData::Uint(_))
            && file_type != PixelType::HALF
            && file_type != PixelType::FLOAT
        {
            err.push_str("Invalid requested_pixel_type.\n");
            return false;
        }
        for y in 0..num_lines as usize {
            // Assume increasing Y.
            let line = pixel_data_size * y * width_size + channel_offset_list[c] * width_size;
            let first = (y + start_y) * x_stride as usize;
            let half = file_type == PixelType::HALF && !matches!(image, ChannelData::Uint(_));
            let bytes = if half { 2 } else { 4 };
            let dst = &mut buf[line..line + bytes * width_size];
            match image {
                // (Whatever they are to be stored as: `check` has refused what has not four
                // bytes a value.)
                ChannelData::Uint(values) => {
                    let src = &values[first..first + width_size];
                    for (to, value) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                        *to = value.to_le_bytes();
                    }
                }
                ChannelData::Half(values) => {
                    let src = &values[first..first + width_size];
                    if file_type == PixelType::FLOAT {
                        for (to, &value) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                            *to = half_to_float(value).to_le_bytes();
                        }
                    } else {
                        for (to, value) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                            *to = value.to_le_bytes();
                        }
                    }
                }
                ChannelData::Float(values) => {
                    let src = &values[first..first + width_size];
                    if file_type == PixelType::HALF {
                        for (to, &value) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                            *to = float_to_half(value).to_le_bytes();
                        }
                    } else {
                        for (to, value) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                            *to = value.to_le_bytes();
                        }
                    }
                }
            }
        }
    }

    if compression_type == Compression::NONE {
        // 4 byte: scan line
        // 4 byte: data size
        // ~     : pixel data(uncompressed)
        out_data.extend_from_slice(&buf);
    } else if compression_type == Compression::ZIPS || compression_type == Compression::ZIP {
        // 4 byte: scan line
        // 4 byte: data size
        // ~     : pixel data(compressed)
        out_data.extend_from_slice(&compress_zip(&buf));
    } else if compression_type == Compression::RLE {
        // (buf.size() * 3) / 2 would be enough.
        let Some(block) = compress_rle(&buf) else {
            err.push_str("RLE compresssion failed.\n");
            return false;
        };
        out_data.extend_from_slice(&block);
    } else if compression_type == Compression::PIZ {
        let channel_sizes: Vec<usize> = file_types
            .iter()
            .map(|&file_type| if file_type == PixelType::HALF { 1 } else { 2 })
            .collect();
        out_data.extend_from_slice(&compress_piz(&buf, &channel_sizes, width, num_lines));
    } else if compression_type == Compression::ZFP {
        // 4 byte: scan line
        // 4 byte: data size
        // ~     : pixel data(compressed)
        // (`check` has refused what the original would not compress.)
        let Ok(block) = compress_zfp(&buf, width, num_lines, file_types.len(), zfp) else {
            return false;
        };
        out_data.extend_from_slice(&block);
    } else if compression_type == Compression::PXR24 {
        let block = compress_pxr24(&buf, width, num_lines, file_types);
        // (The format's rule, which the original has for its other compressions and not for
        // this one: a block that does not come out smaller is stored as it is.)
        out_data.extend_from_slice(if !as_tinyexr && block.len() >= buf.len() {
            &buf
        } else {
            &block
        });
    } else if compression_type == Compression::B44 || compression_type == Compression::B44A {
        let is_b44a = compression_type == Compression::B44A;
        // B44 expects channel-sequential layout. Rearrange from line-interleaved to
        // channel-sequential.
        let mut seq_buf = vec![0u8; buf_size];
        let mut seq_p = 0usize;
        for (c, &file_type) in file_types.iter().enumerate() {
            let ch_size = if file_type == PixelType::HALF { 2 } else { 4 };
            for y in 0..num_lines as usize {
                let src = y * pixel_data_size * width_size + channel_offset_list[c] * width_size;
                let row_bytes = width_size * ch_size;
                seq_buf[seq_p..seq_p + row_bytes].copy_from_slice(&buf[src..src + row_bytes]);
                seq_p += row_bytes;
            }
        }
        let block = compress_b44(&seq_buf, width, num_lines, file_types, is_b44a);
        // (The format's rule again; for B44 this is every block of an image without a
        // channel of halves.)
        out_data.extend_from_slice(if !as_tinyexr && block.len() >= buf.len() {
            &buf
        } else {
            &block
        });
    } else if !as_tinyexr
        && (compression_type == Compression::DWAA || compression_type == Compression::DWAB)
    {
        // (Not in the original. A block that does not come out smaller is stored as it is.)
        let channels: Vec<dwa::Channel> = dwa
            .names
            .iter()
            .zip(file_types)
            .map(|(name, &pixel_type)| dwa::Channel {
                name,
                pixel_type: DwaPixelType::from_code(pixel_type.0).expect("a pixel type"),
                linear: false,
            })
            .collect();
        let ac_compression = if dwa.deflate_ac {
            dwa::AcCompression::Deflate
        } else {
            dwa::AcCompression::Huffman
        };
        out_data.extend_from_slice(
            &dwa::compress(
                &buf,
                &channels,
                width_size,
                num_lines as usize,
                dwa.level,
                ac_compression,
            )
            .expect("a block of the channels' size"),
        );
    } else {
        return false;
    }

    true
}

/// What a block of DWAA or DWAB needs of the header: the channels' names, the level, and
/// whether its AC coefficients are compressed by ZIP, as OpenEXR has them for the tiles of
/// DWAA (by Huffman coding otherwise).
pub(super) struct DwaParam {
    names: Vec<String>,
    level: f32,
    deflate_ac: bool,
}

impl DwaParam {
    fn new(exr_header: &ExrHeader, tiled: bool) -> DwaParam {
        let names = exr_header
            .channels
            .iter()
            .map(|channel| channel.name.split('\0').next().unwrap_or("").to_string())
            .collect();
        DwaParam {
            names,
            level: dwa_level(&exr_header.custom_attributes),
            deflate_ac: tiled && exr_header.compression_type == Compression::DWAA,
        }
    }
}

/// The `dwaCompressionLevel` of a header's attributes (a `float`), or OpenEXR's 45.
fn dwa_level(attributes: &[Attribute]) -> f32 {
    attributes
        .iter()
        .find(|attribute| {
            attribute.name == "dwaCompressionLevel"
                && attribute.type_name == "float"
                && attribute.value.len() == 4
        })
        .map_or(45.0, |attribute| {
            f32::from_le_bytes([
                attribute.value[0],
                attribute.value[1],
                attribute.value[2],
                attribute.value[3],
            ])
        })
}

/// The attribute that sets the level of DWAA and DWAB compression (`dwaCompressionLevel`),
/// for a header's `custom_attributes`: OpenEXR's "base error" times 100000, from 0 to
/// 6550400000. With 0 nothing is lost to the quantization (but some to the rest); OpenEXR's
/// default is 45.
pub fn dwa_compression_level_attribute(level: f32) -> Attribute {
    Attribute {
        name: "dwaCompressionLevel".to_string(),
        type_name: "float".to_string(),
        value: level.to_le_bytes().to_vec(),
    }
}

/// What the port refuses to write with DWAA or DWAB compression, as OpenEXR refuses it.
pub const DWA_INVALID_LEVEL: &str = "Invalid dwa quality level specified";

/// `EncodeTiledLevel`: the tiles of one level, each as a block in `data_list` from
/// `start_index` on.
#[allow(clippy::too_many_arguments)]
fn encode_tiled_level(
    level_image: &ExrImage,
    exr_header: &ExrHeader,
    data_list: &mut [Vec<u8>],
    start_index: usize,
    num_x_tiles: i32,
    num_y_tiles: i32,
    channel_offset_list: &[usize],
    pixel_data_size: usize,
    zfp: &ZfpParam,
    dwa: &DwaParam,
    err: &mut String,
    as_tinyexr: bool,
) -> ErrorCode {
    let num_tiles = num_x_tiles.wrapping_mul(num_y_tiles);
    if num_tiles as i64 != level_image.tiles.len() as i64 {
        err.push_str("Invalid number of tiles in argument.\n");
        return ErrorCode::INVALID_ARGUMENT;
    }

    if (exr_header.tile_size_x > level_image.width || exr_header.tile_size_y > level_image.height)
        && level_image.level_x == 0
        && level_image.level_y == 0
    {
        err.push_str("Failed to encode tile data.\n");
        return ErrorCode::INVALID_DATA;
    }

    let mut invalid_data = false;
    for i in 0..num_tiles {
        let tile_idx = i as usize;
        let data_idx = tile_idx + start_index;

        let x_tile = i % num_x_tiles;
        let y_tile = i / num_x_tiles;

        let tile = &level_image.tiles[tile_idx];

        // 5 ints: the tile's place, its level, and its size in the file.
        let mut data = vec![0u8; 5 * 4];
        let data_header_size = data.len();
        let ret = encode_pixel_data(
            &mut data,
            &tile.images,
            exr_header.compression_type,
            tile.width,
            // stride
            exr_header.tile_size_x,
            // line_no
            0,
            tile.height,
            pixel_data_size,
            &exr_header.requested_pixel_types,
            channel_offset_list,
            zfp,
            dwa,
            err,
            as_tinyexr,
        );
        if !ret {
            invalid_data = true;
            continue;
        }
        if data.len() <= data_header_size {
            invalid_data = true;
            continue;
        }

        let data_len = (data.len() - data_header_size) as i32;
        // x_tile, y_tile, level_x, level_y, data_len
        data[0..4].copy_from_slice(&x_tile.to_le_bytes());
        data[4..8].copy_from_slice(&y_tile.to_le_bytes());
        data[8..12].copy_from_slice(&level_image.level_x.to_le_bytes());
        data[12..16].copy_from_slice(&level_image.level_y.to_le_bytes());
        data[16..20].copy_from_slice(&data_len.to_le_bytes());
        data_list[data_idx] = data;
    }

    if invalid_data {
        err.push_str("Failed to encode tile data.\n");
        return ErrorCode::INVALID_DATA;
    }
    ErrorCode::SUCCESS
}

/// `EncodeChunk`: every block of the image, in `data_list`, and where each will be in the
/// file, in `offset_data`. `total_size` comes back as the size of the file.
#[allow(clippy::too_many_arguments)]
fn encode_chunk(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
    num_blocks: i32,
    chunk_offset: u64,
    is_multipart: bool,
    offset_data: &mut OffsetData,
    data_list: &mut Vec<Vec<u8>>,
    total_size: &mut u64,
    err: &mut String,
    as_tinyexr: bool,
) -> ErrorCode {
    let num_scanlines = num_scanline_blocks(exr_header.compression_type, as_tinyexr);

    data_list.clear();
    data_list.resize(num_blocks as usize, Vec::new());

    let mut channel_offset_list = Vec::with_capacity(exr_header.channels.len());

    let mut pixel_data_size = 0usize;
    {
        let mut channel_offset = 0usize;
        for &file_type in &exr_header.requested_pixel_types {
            channel_offset_list.push(channel_offset);
            if file_type == PixelType::HALF {
                pixel_data_size += 2;
                channel_offset += 2;
            } else if file_type == PixelType::FLOAT || file_type == PixelType::UINT {
                pixel_data_size += 4;
                channel_offset += 4;
            } else {
                err.push_str("Invalid requested_pixel_type.\n");
                return ErrorCode::INVALID_DATA;
            }
        }
    }

    // Use ZFP compression parameter from custom attributes(if such a parameter exists)
    let zfp = zfp_param_for_writing(&exr_header.custom_attributes);
    let dwa = DwaParam::new(exr_header, !exr_image.tiles.is_empty());

    let mut offset = chunk_offset;
    // (In a file of many parts, each block has its part's number before it.)
    let doffset = if is_multipart { 4u64 } else { 0 };

    if !exr_image.tiles.is_empty() {
        let mut level_image = Some(exr_image);
        let mut block_idx = 0usize;
        let num_levels = if exr_header.tile_level_mode != TileLevelMode::RIPMAP_LEVELS {
            offset_data.num_x_levels
        } else {
            offset_data
                .num_x_levels
                .wrapping_mul(offset_data.num_y_levels)
        };
        for level in 0..num_levels {
            let Some(image) = level_image else {
                err.push_str("Invalid number of tiled levels for EncodeChunk\n");
                return ErrorCode::INVALID_DATA;
            };

            let level_index_from_image = level_index(
                image.level_x,
                image.level_y,
                exr_header.tile_level_mode,
                offset_data.num_x_levels,
            );
            if level_index_from_image < 0 {
                err.push_str("Invalid tile level mode\n");
                return ErrorCode::INVALID_DATA;
            }

            if level_index_from_image != level {
                err.push_str("Incorrect level ordering in tiled image\n");
                return ErrorCode::INVALID_DATA;
            }
            let num_y_tiles = offset_data.offsets[level as usize].len() as i32;
            if num_y_tiles <= 0 {
                err.push_str("Invalid Y tile size\n");
                return ErrorCode::INVALID_DATA;
            }

            let num_x_tiles = offset_data.offsets[level as usize][0].len() as i32;
            if num_x_tiles <= 0 {
                err.push_str("Invalid X tile size\n");
                return ErrorCode::INVALID_DATA;
            }

            let mut e = String::new();
            let ret = encode_tiled_level(
                image,
                exr_header,
                data_list,
                block_idx,
                num_x_tiles,
                num_y_tiles,
                &channel_offset_list,
                pixel_data_size,
                &zfp,
                &dwa,
                &mut e,
                as_tinyexr,
            );
            if ret != ErrorCode::SUCCESS {
                if !e.is_empty() {
                    err.push_str(&e);
                }
                return ret;
            }

            for j in 0..num_y_tiles as usize {
                for i in 0..num_x_tiles as usize {
                    offset_data.offsets[level as usize][j][i] = offset;
                    offset += data_list[block_idx].len() as u64 + doffset;
                    block_idx += 1;
                }
            }
            level_image = image.next_level.as_deref();
        }
        if block_idx as i64 != num_blocks as i64 {
            return ErrorCode::INVALID_DATA;
        }
        *total_size = offset;
    } else {
        // scanlines
        let mut invalid_data = false;

        for i in 0..num_blocks {
            let start_y = num_scanlines * i;
            let end_y = (num_scanlines * (i + 1)).min(exr_image.height);
            let num_lines = end_y - start_y;

            // 2 ints: the block's first line and its size in the file.
            let mut data = vec![0u8; 2 * 4];
            let data_header_size = data.len();

            let ret = encode_pixel_data(
                &mut data,
                &exr_image.images,
                exr_header.compression_type,
                exr_image.width,
                exr_image.width,
                // start_y
                start_y,
                num_lines,
                pixel_data_size,
                &exr_header.requested_pixel_types,
                &channel_offset_list,
                &zfp,
                &dwa,
                err,
                as_tinyexr,
            );
            if !ret {
                invalid_data = true;
                continue;
            }
            if data.len() <= data_header_size {
                invalid_data = true;
                continue;
            }
            let data_len = (data.len() - data_header_size) as i32;
            data[0..4].copy_from_slice(&start_y.to_le_bytes());
            data[4..8].copy_from_slice(&data_len.to_le_bytes());
            data_list[i as usize] = data;
        }

        if invalid_data {
            err.push_str("Failed to encode scanline data.\n");
            return ErrorCode::INVALID_DATA;
        }

        let offsets = &mut offset_data.offsets[0][0];
        for (slot, data) in offsets.iter_mut().zip(data_list.iter()) {
            *slot = offset;
            offset += data.len() as u64 + doffset;
        }

        *total_size = offset;
    }
    ErrorCode::SUCCESS
}

/// What the original would read past its memory for, or throw on, in what it is handed.
/// (It is handed pointers and trusts the header for what is behind them.)
fn check(exr_image: &ExrImage, exr_header: &ExrHeader, as_tinyexr: bool) -> Result<(), ExrError> {
    let refuse = |message: &str| {
        Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: message.to_string(),
        })
    };
    // (OpenEXR's bounds for the level of DWAA and DWAB.)
    if !as_tinyexr
        && (exr_header.compression_type == Compression::DWAA
            || exr_header.compression_type == Compression::DWAB)
    {
        let level = dwa_level(&exr_header.custom_attributes);
        if !(0.0..=65504.0 * 100000.0).contains(&level) {
            return refuse(DWA_INVALID_LEVEL);
        }
    }
    let num_channels = exr_header.channels.len();
    if num_channels == 0 {
        return refuse("An image needs a channel.");
    }
    if exr_header.pixel_types.len() != num_channels
        || exr_header.requested_pixel_types.len() != num_channels
    {
        return refuse("pixel_types and requested_pixel_types need a type for each channel.");
    }
    // The values of every channel are there, of the type the header says, with a type to
    // store them as that the original has code for.
    let check_images = |images: &[ChannelData], needed: usize| {
        if images.len() != num_channels {
            return refuse("An image or a tile needs the values of each channel.");
        }
        for (c, image) in images.iter().enumerate() {
            let (pixel_type, len) = match image {
                ChannelData::Uint(values) => (PixelType::UINT, values.len()),
                ChannelData::Half(values) => (PixelType::HALF, values.len()),
                ChannelData::Float(values) => (PixelType::FLOAT, values.len()),
            };
            if exr_header.pixel_types[c] != pixel_type {
                return refuse("pixel_types does not say what a channel's values are.");
            }
            if pixel_type == PixelType::UINT
                && exr_header.requested_pixel_types[c] == PixelType::HALF
            {
                return refuse("A channel of integers cannot be stored as halves.");
            }
            if len < needed {
                return refuse("A channel has fewer values than the image or the tile has pixels.");
            }
        }
        Ok(())
    };
    // ZFP: blocks and tiles of whole blocks of four by four, and settings that do not ask for
    // more than there may be. (The original looks at neither; it writes a length it never set
    // for the first, and asks for the memory for the second.)
    let zfp = (exr_header.compression_type == Compression::ZFP)
        .then(|| zfp_param_for_writing(&exr_header.custom_attributes));
    let check_zfp = |sizes: &[(i32, i32)]| {
        let Some(zfp) = &zfp else { return Ok(()) };
        if sizes
            .iter()
            .any(|&(width, lines)| width % 4 != 0 || lines % 4 != 0)
        {
            return refuse(ZFP_NOT_MULTIPLE_OF_FOUR);
        }
        if sizes
            .iter()
            .any(|&(width, lines)| zfp_stream_size(zfp, width, lines, num_channels).is_none())
        {
            return refuse(ZFP_TOO_LARGE);
        }
        Ok(())
    };
    if exr_image.tiles.is_empty() {
        if exr_image.width < 1 || exr_image.height < 1 {
            return refuse("An image needs a width and a height.");
        }
        check_images(
            &exr_image.images,
            exr_image.width as usize * exr_image.height as usize,
        )?;
        let lines = num_scanline_blocks(exr_header.compression_type, as_tinyexr);
        let sizes: Vec<(i32, i32)> = (0..exr_image.height)
            .step_by(lines as usize)
            .map(|start| (exr_image.width, lines.min(exr_image.height - start)))
            .collect();
        return check_zfp(&sizes);
    }
    if exr_header.tile_size_x < 1 || exr_header.tile_size_y < 1 {
        return refuse("A tiled image needs a tile size.");
    }
    let mut level = Some(exr_image);
    while let Some(image) = level {
        for tile in &image.tiles {
            let ExrTile { width, height, .. } = *tile;
            if width < 1 || height < 1 {
                return refuse("A tile needs a width and a height.");
            }
            check_images(
                &tile.images,
                (height as usize - 1) * exr_header.tile_size_x as usize + width as usize,
            )?;
        }
        level = image.next_level.as_deref();
    }
    let mut sizes = Vec::new();
    let mut level = Some(exr_image);
    while let Some(image) = level {
        sizes.extend(image.tiles.iter().map(|tile| (tile.width, tile.height)));
        level = image.next_level.as_deref();
    }
    check_zfp(&sizes)
}

/// The message for a block or tile that ZFP compression cannot write because its width or
/// height is not a multiple of four.
pub const ZFP_NOT_MULTIPLE_OF_FOUR: &str =
    "ZFP compression needs blocks and tiles whose width and height are multiples of four.";
/// The message for ZFP settings that would need more than a gigabyte for a block.
pub const ZFP_TOO_LARGE: &str =
    "ZFP compression with these settings needs more than a gigabyte for a block.";

/// `SaveEXRImageToMemory`: an image as the bytes of a `.exr` file of one part.
///
/// `exr_header` says how: its `channels` name the channels (in the order of their names, as
/// the format wants them), `pixel_types` say what the values of each are, as `exr_image` has
/// them, and `requested_pixel_types` what each is stored as (halves and floats as either,
/// integers as integers); `compression_type` is one of the `COMPRESSIONTYPE_*` but ZFP, and
/// `custom_attributes` are written after the attributes every file has. The size is the
/// image's, and the data window starts at (0, 0); line order, pixel aspect ratio and screen
/// window are written as 0, 1 and the usual ones whatever the header says, as the original
/// writes them.
///
/// An image with `tiles` is written tiled: `tile_size_x`, `tile_size_y`, `tile_level_mode`
/// and `tile_rounding_mode` of the header say how, and its `data_window` has to be the
/// image's. Each level is an `ExrImage` with its tiles in order, row after row, and the next
/// level in `next_level` ([`tile_levels`] says which levels and how many tiles).
///
/// Where the original answers a size of 0 and a message, the error's code is
/// `ErrorCode::SERIALIZATION_FAILED`. What the original does not look at and would read past its
/// memory for (a channel with too few values, a type that is not the values' type) is
/// refused with `ErrorCode::INVALID_ARGUMENT` and a message of the port's own.
///
///
/// The file is the one tinyexr writes, byte for byte, but for PXR24, B44 and B44A blocks
/// that do not come out smaller than their pixels: those are stored as the pixels, as the
/// format has it and OpenEXR writes them, and as tinyexr itself does for RLE, ZIP and PIZ.
/// [`save_image_as_tinyexr`] writes them as tinyexr does.
///
/// [`tile_levels`]: super::tile_levels
pub fn save_image(exr_image: &ExrImage, exr_header: &ExrHeader) -> Result<Vec<u8>, ExrError> {
    save_image_with(exr_image, exr_header, false)
}

/// `SaveEXRImageToMemory` exactly as tinyexr has it: the same file, byte for byte, for every
/// image. For PXR24, B44 and B44A that includes blocks that are not smaller than their
/// pixels, which a reader that follows the format takes for the pixels themselves and reads
/// as other values: every block of a B44 image without a channel of halves, and the blocks
/// of a few pixels of any image. ([`load_image_as_tinyexr`] reads them as
/// tinyexr does.)
///
/// [`load_image_as_tinyexr`]: super::load_image_as_tinyexr
pub fn save_image_as_tinyexr(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
) -> Result<Vec<u8>, ExrError> {
    save_image_with(exr_image, exr_header, true)
}

fn save_image_with(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
    as_tinyexr: bool,
) -> Result<Vec<u8>, ExrError> {
    save_parts(&[(exr_image, exr_header)], as_tinyexr)
}

/// `SaveEXRMultipartImageToMemory`: images as the parts of one `.exr` file, each with its
/// header, as [`save_image`] writes one. Each part needs a `name`, and no two
/// the same; the size of the first part is every part's display window. With one part the
/// file is the one [`save_image`] writes.
pub fn save_multipart_images(parts: &[(&ExrImage, &ExrHeader)]) -> Result<Vec<u8>, ExrError> {
    save_multipart(parts, false)
}

/// `SaveEXRMultipartImageToMemory` exactly as tinyexr has it
/// ([`save_image_as_tinyexr`] says where that differs).
pub fn save_multipart_images_as_tinyexr(
    parts: &[(&ExrImage, &ExrHeader)],
) -> Result<Vec<u8>, ExrError> {
    save_multipart(parts, true)
}

fn save_multipart(
    parts: &[(&ExrImage, &ExrHeader)],
    as_tinyexr: bool,
) -> Result<Vec<u8>, ExrError> {
    if parts.is_empty() {
        return Err(ExrError {
            code: ErrorCode::SERIALIZATION_FAILED,
            message: "Invalid argument for SaveEXRNPartImageToMemory".to_string(),
        });
    }
    save_parts(parts, as_tinyexr)
}

/// `SaveEXRNPartImageToMemory`.
fn save_parts(parts: &[(&ExrImage, &ExrHeader)], as_tinyexr: bool) -> Result<Vec<u8>, ExrError> {
    let fail = |code: ErrorCode, message: &str| {
        Err(ExrError {
            code,
            message: message.to_string(),
        })
    };
    let num_parts = parts.len();

    for &(_, exr_header) in parts {
        if exr_header.compression_type.0 < 0 {
            return fail(
                ErrorCode::SERIALIZATION_FAILED,
                "Invalid argument for SaveEXRNPartImageToMemory",
            );
        }
        if exr_header.compression_type == Compression::ZFP {
            // All channels must be fp32.
            // No fp16 support in ZFP atm(as of 2023 June)
            // (`check` refuses a header without a type for each channel.)
            if exr_header
                .requested_pixel_types
                .iter()
                .take(exr_header.channels.len())
                .any(|&file_type| file_type != PixelType::FLOAT)
            {
                return fail(
                    ErrorCode::SERIALIZATION_FAILED,
                    "Pixel type must be FLOAT for ZFP compression",
                );
            }
        }
    }
    for &(exr_image, exr_header) in parts {
        check(exr_image, exr_header, as_tinyexr)?;
    }

    let mut memory = Vec::new();

    // Header
    memory.extend_from_slice(&[0x76, 0x2f, 0x31, 0x01]);

    // Version
    // using value from the first header
    let long_name = parts[0].1.long_name;
    {
        let mut marker = [2u8, 0, 0, 0];
        // @todo
        // tiled
        if num_parts == 1 && !parts[0].0.tiles.is_empty() {
            marker[1] |= 0x2;
        }
        // long_name
        if long_name != 0 {
            marker[1] |= 0x4;
        }
        // multipart
        if num_parts > 1 {
            marker[1] |= 0x10;
        }
        memory.extend_from_slice(&marker);
    }

    let mut total_chunk_count = 0i32;
    let mut chunk_count = vec![0i32; num_parts];
    let mut offset_data: Vec<OffsetData> = (0..num_parts).map(|_| OffsetData::default()).collect();
    for (i, &(exr_image, exr_header)) in parts.iter().enumerate() {
        if exr_image.tiles.is_empty() {
            let num_scanlines = num_scanline_blocks(exr_header.compression_type, as_tinyexr);
            chunk_count[i] = (exr_image.height + num_scanlines - 1) / num_scanlines;
            // `InitSingleResolutionOffsets`.
            offset_data[i].offsets = vec![vec![vec![0u64; chunk_count[i] as usize]]];
            offset_data[i].num_x_levels = 1;
            offset_data[i].num_y_levels = 1;
        } else {
            let Some((num_x_tiles, num_y_tiles)) = precalculate_tile_info(exr_header) else {
                return fail(ErrorCode::INVALID_DATA, "Failed to precalculate Tile info");
            };
            let (shape, ntiles) =
                tile_offsets_shape(&mut offset_data[i], exr_header, &num_x_tiles, &num_y_tiles);
            if ntiles > 0 {
                chunk_count[i] = ntiles;
            } else {
                return fail(ErrorCode::INVALID_DATA, "Failed to compute Tile offsets");
            }
            offset_data[i].offsets = shape
                .iter()
                .map(|&(rows, columns)| vec![vec![0u64; columns]; rows])
                .collect();
        }
        total_chunk_count = total_chunk_count.wrapping_add(chunk_count[i]);
    }

    // Write attributes to memory buffer.
    let mut partnames: Vec<Vec<u8>> = Vec::new();
    for (i, &(exr_image, exr_header)) in parts.iter().enumerate() {
        //channels
        {
            let data = write_channel_info(&exr_header.channels, &exr_header.requested_pixel_types);
            write_attribute_to_memory(&mut memory, "channels", "chlist", &data);
        }

        write_attribute_to_memory(
            &mut memory,
            "compression",
            "compression",
            &[exr_header.compression_type.0 as u8],
        );

        {
            let mut data = Vec::with_capacity(16);
            for value in [
                0,
                0,
                exr_image.width.wrapping_sub(1),
                exr_image.height.wrapping_sub(1),
            ] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            write_attribute_to_memory(&mut memory, "dataWindow", "box2i", &data);

            let first = parts[0].0;
            let mut data0 = Vec::with_capacity(16);
            for value in [
                0,
                0,
                first.width.wrapping_sub(1),
                first.height.wrapping_sub(1),
            ] {
                data0.extend_from_slice(&value.to_le_bytes());
            }
            // Note: must be the same across parts (currently, using value from the first header)
            write_attribute_to_memory(&mut memory, "displayWindow", "box2i", &data0);
        }

        // @fixme: the original writes increasing Y whatever the header says.
        write_attribute_to_memory(&mut memory, "lineOrder", "lineOrder", &[0]);

        // Note: must be the same across parts
        write_attribute_to_memory(
            &mut memory,
            "pixelAspectRatio",
            "float",
            &1.0f32.to_le_bytes(),
        );

        write_attribute_to_memory(&mut memory, "screenWindowCenter", "v2f", &[0; 8]);

        write_attribute_to_memory(
            &mut memory,
            "screenWindowWidth",
            "float",
            &1.0f32.to_le_bytes(),
        );

        if !exr_image.tiles.is_empty() {
            let mut tile_mode = (exr_header.tile_level_mode.0 & 0x3) as u8;
            if exr_header.tile_rounding_mode != TileRoundingMode::ROUND_DOWN {
                tile_mode |= 1 << 4;
            }
            let mut data = Vec::with_capacity(9);
            data.extend_from_slice(&exr_header.tile_size_x.to_le_bytes());
            data.extend_from_slice(&exr_header.tile_size_y.to_le_bytes());
            data.push(tile_mode);
            write_attribute_to_memory(&mut memory, "tiles", "tiledesc", &data);
        }

        // must be present for multi-part files - according to spec.
        if num_parts > 1 {
            // name (the original's is a C string of at most 255 bytes)
            {
                let name = exr_header.name.as_bytes();
                let name = &name[..name
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(name.len())
                    .min(255)];
                if !name.is_empty() {
                    if !partnames.iter().any(|seen| seen == name) {
                        partnames.push(name.to_vec());
                    }
                    if partnames.len() != i + 1 {
                        return fail(
                            ErrorCode::SERIALIZATION_FAILED,
                            "'name' attributes must be unique for a multi-part file",
                        );
                    }
                    write_attribute_bytes(&mut memory, b"name", b"string", name);
                } else {
                    return fail(
                        ErrorCode::SERIALIZATION_FAILED,
                        "Invalid 'name' attribute for a multi-part file",
                    );
                }
            }
            // type
            {
                let type_name = if !exr_image.tiles.is_empty() {
                    "tiledimage"
                } else {
                    "scanlineimage"
                };
                write_attribute_to_memory(&mut memory, "type", "string", type_name.as_bytes());
            }
            // chunkCount
            write_attribute_to_memory(
                &mut memory,
                "chunkCount",
                "int",
                &chunk_count[i].to_le_bytes(),
            );
        }

        // Custom attributes
        for attribute in &exr_header.custom_attributes {
            write_attribute_to_memory(
                &mut memory,
                &attribute.name,
                &attribute.type_name,
                &attribute.value,
            );
        }

        // end of header
        memory.push(0);
    }
    if num_parts > 1 {
        // end of header list
        memory.push(0);
    }

    let mut chunk_offset = memory.len() as u64 + total_chunk_count as i64 as u64 * 8;

    let mut total_size = 0u64;
    let mut data_lists: Vec<Vec<Vec<u8>>> = vec![Vec::new(); num_parts];
    for (i, &(exr_image, exr_header)) in parts.iter().enumerate() {
        let mut e = String::new();
        let ret = encode_chunk(
            exr_image,
            exr_header,
            chunk_count[i],
            chunk_offset,
            num_parts > 1,
            &mut offset_data[i],
            &mut data_lists[i],
            &mut total_size,
            &mut e,
            as_tinyexr,
        );
        if ret != ErrorCode::SUCCESS {
            return Err(ExrError {
                code: ErrorCode::SERIALIZATION_FAILED,
                message: e,
            });
        }
        chunk_offset = total_size;
    }

    // Allocating required memory
    if total_size == 0 {
        // something went wrong
        return fail(ErrorCode::INVALID_DATA, "Output memory size is zero");
    }

    // Writing header, then the offsets, then the blocks.
    for data in &offset_data {
        for offset in data.offsets.iter().flatten().flatten() {
            memory.extend_from_slice(&offset.to_le_bytes());
        }
    }
    for (i, data_list) in data_lists.iter().enumerate() {
        for data in data_list {
            if num_parts > 1 {
                memory.extend_from_slice(&(i as u32).to_le_bytes());
            }
            memory.extend_from_slice(data);
        }
    }

    if memory.len() as u64 != total_size {
        return fail(ErrorCode::INVALID_DATA, "Corrupted Part image chunk data.");
    }

    Ok(memory)
}

/// `SaveEXRToMemory`: an image of floats as the bytes of a `.exr` file. `data` holds
/// `components` floats a pixel, line by line: one (which is written as a channel named `A`,
/// as the original names it), three (red, green and blue) or four (and alpha). They are
/// stored as halves if `save_as_fp16`, as floats if not; compressed with ZIP, or not at all
/// when the image is less than sixteen pixels each way.
pub fn save_floats(
    data: &[f32],
    width: i32,
    height: i32,
    components: i32,
    save_as_fp16: bool,
) -> Result<Vec<u8>, ExrError> {
    if !(components == 1 || components == 3 || components == 4) {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: format!("Unsupported component value : {components}\n"),
        });
    }
    // (The original trusts these, and reads past `data` or stops when they are wrong.)
    if width < 1
        || height < 1
        || (data.len() as u64) < width as u64 * height as u64 * components as u64
    {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: "The image has fewer floats than its width, height and components say."
                .to_string(),
        });
    }

    let mut header = ExrHeader {
        compression_type: if width < 16 && height < 16 {
            Compression::NONE
        } else {
            Compression::ZIP
        },
        ..ExrHeader::default()
    };

    let pixels = width as usize * height as usize;
    let components_size = components as usize;
    // The channels one by one, in the order of the pixel's floats.
    let split: Vec<Vec<f32>> = (0..components_size)
        .map(|k| (0..pixels).map(|i| data[components_size * i + k]).collect())
        .collect();

    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    let names: &[&str] = match components {
        4 => &["A", "B", "G", "R"],
        3 => &["B", "G", "R"],
        _ => &["A"],
    };
    let image = ExrImage {
        images: split.into_iter().rev().map(ChannelData::Float).collect(),
        width,
        height,
        num_channels: components,
        ..ExrImage::default()
    };

    header.channels = names
        .iter()
        .map(|name| ChannelInfo {
            name: name.to_string(),
            pixel_type: PixelType::FLOAT,
            x_sampling: 1,
            y_sampling: 1,
            p_linear: 0,
        })
        .collect();
    // pixel type of input image
    header.pixel_types = vec![PixelType::FLOAT; components_size];
    // pixel type of output image to be stored in .EXR
    header.requested_pixel_types = vec![
        if save_as_fp16 {
            PixelType::HALF
        } else {
            PixelType::FLOAT
        };
        components_size
    ];

    let memory = save_image(&image, &header)?;
    if memory.len() > i32::MAX as usize {
        return Err(ExrError {
            code: ErrorCode::DATA_TOO_LARGE,
            message: String::new(),
        });
    }
    Ok(memory)
}
