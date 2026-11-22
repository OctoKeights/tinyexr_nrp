//! Writing a file: a port of `SaveEXRImageToMemory` and `SaveEXRMultipartImageToMemory`
//! (`SaveEXRNPartImageToMemory`), `EncodeChunk`, `EncodeTiledLevel`, `EncodePixelData` and
//! `SaveEXRToMemory` of tinyexr v3.4.0 (`tinyexr.h`).

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use super::b44::compress_b44;
use super::chunk::{
    OffsetData, effective_threads, level_index, num_scanline_blocks, precalculate_tile_info,
    tile_offsets_shape,
};
use super::decode::ChannelData;
use super::piz::{PizScratch, compress_piz_into};
use super::pxr24::compress_pxr24_into;
use super::zfp::{ZfpParam, zfp_param_for_writing};
#[cfg(feature = "zfp")]
use super::zfp::{compress_zfp, zfp_stream_size};
use super::zip_rle::{compress_rle_into, compress_zip_into};
use super::{
    Attribute, ChannelInfo, Compression, ErrorCode, ExrError, ExrHeader, ExrImage, ExrTile,
    PixelType, TileLevelMode, TileRoundingMode, float_to_half, half_to_float,
};
#[cfg(feature = "dwa")]
use openexr_nrp::{PixelType as DwaPixelType, dwa};

/// Per-thread reusable scratch buffers and codec contexts across chunks.
#[derive(Default)]
struct EncodeScratch {
    buf: Vec<u8>,
    tmp_buf: Vec<u8>,
    aux_buf: Vec<u8>,
    zlib: Option<miniz_nrp::Compressor>,
    piz: PizScratch,
    #[cfg(feature = "dwa")]
    dwa: dwa::Context,
}

/// `WriteAttributeToMemory`.
fn write_attribute_to_memory(out: &mut Vec<u8>, name: &str, type_name: &str, data: &[u8]) {
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
/// stored as. In non-`_as_tinyexr` mode, preserves `ChannelInfo::p_linear` (`0` or `1`).
fn write_channel_info(
    channels: &[ChannelInfo],
    file_types: &[PixelType],
    as_tinyexr: bool,
) -> Vec<u8> {
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
        let p_linear = if as_tinyexr {
            0
        } else {
            u8::from(channel.p_linear != 0)
        };
        data.extend_from_slice(&[p_linear, 0, 0, 0]);
        data.extend_from_slice(&1i32.to_le_bytes());
        data.extend_from_slice(&1i32.to_le_bytes());
    }
    data.push(0);
    data
}

/// `EncodePixelData`: `num_lines` lines of `width` pixels from `line_no` on, taken from
/// `images`, as a block appended to `out_data`, reusing `scratch` across chunks.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    as_tinyexr: bool,
    scratch: &mut EncodeScratch,
) -> Result<(), &'static str> {
    let width_size = width as usize;
    let buf_size = width_size * num_lines as usize * pixel_data_size;
    scratch.buf.clear();
    scratch.buf.resize(buf_size, 0);

    let start_y = line_no as usize;
    for (c, image) in images.iter().enumerate() {
        let file_type = file_types[c];
        if !matches!(image, ChannelData::Uint(_))
            && file_type != PixelType::HALF
            && file_type != PixelType::FLOAT
        {
            return Err("Invalid requested_pixel_type.\n");
        }
        for y in 0..num_lines as usize {
            let line = pixel_data_size * y * width_size + channel_offset_list[c] * width_size;
            let first = (y + start_y) * x_stride as usize;
            let half = file_type == PixelType::HALF && !matches!(image, ChannelData::Uint(_));
            let bytes = if half { 2 } else { 4 };
            let dst = &mut scratch.buf[line..line + bytes * width_size];
            match image {
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
        out_data.extend_from_slice(&scratch.buf);
    } else if compression_type == Compression::ZIPS || compression_type == Compression::ZIP {
        let compressor = scratch
            .zlib
            .get_or_insert_with(miniz_nrp::Compressor::default);
        compress_zip_into(&scratch.buf, &mut scratch.tmp_buf, compressor, out_data);
    } else if compression_type == Compression::RLE {
        if !compress_rle_into(
            &scratch.buf,
            &mut scratch.tmp_buf,
            &mut scratch.aux_buf,
            out_data,
        ) {
            return Err("RLE compresssion failed.\n");
        }
    } else if compression_type == Compression::PIZ {
        let channel_sizes: Vec<usize> = file_types
            .iter()
            .map(|&file_type| if file_type == PixelType::HALF { 1 } else { 2 })
            .collect();
        compress_piz_into(
            &scratch.buf,
            &channel_sizes,
            width,
            num_lines,
            &mut scratch.piz,
            out_data,
        );
    } else if compression_type == Compression::ZFP {
        #[cfg(not(feature = "zfp"))]
        {
            let _ = zfp;
            return Err("");
        }
        #[cfg(feature = "zfp")]
        {
            let Ok(block) = compress_zfp(&scratch.buf, width, num_lines, file_types.len(), zfp)
            else {
                return Err("");
            };
            out_data.extend_from_slice(&block);
        }
    } else if compression_type == Compression::PXR24 {
        let compressor = scratch
            .zlib
            .get_or_insert_with(miniz_nrp::Compressor::default);
        compress_pxr24_into(
            &scratch.buf,
            width,
            num_lines,
            file_types,
            as_tinyexr,
            &mut scratch.tmp_buf,
            compressor,
            out_data,
        );
    } else if compression_type == Compression::B44 || compression_type == Compression::B44A {
        let is_b44a = compression_type == Compression::B44A;
        scratch.aux_buf.clear();
        scratch.aux_buf.resize(buf_size, 0);
        let mut seq_p = 0usize;
        for (c, &file_type) in file_types.iter().enumerate() {
            let ch_size = if file_type == PixelType::HALF { 2 } else { 4 };
            for y in 0..num_lines as usize {
                let src = y * pixel_data_size * width_size + channel_offset_list[c] * width_size;
                let row_bytes = width_size * ch_size;
                scratch.aux_buf[seq_p..seq_p + row_bytes]
                    .copy_from_slice(&scratch.buf[src..src + row_bytes]);
                seq_p += row_bytes;
            }
        }
        let block = compress_b44(&scratch.aux_buf, width, num_lines, file_types, is_b44a);
        out_data.extend_from_slice(if !as_tinyexr && block.len() >= scratch.buf.len() {
            &scratch.buf
        } else {
            &block
        });
    } else if !as_tinyexr
        && (compression_type == Compression::DWAA || compression_type == Compression::DWAB)
    {
        #[cfg(not(feature = "dwa"))]
        {
            let _ = dwa;
            return Err("");
        }
        #[cfg(feature = "dwa")]
        {
            let channels: Vec<dwa::Channel> = dwa
                .names
                .iter()
                .zip(&dwa.linear)
                .zip(file_types)
                .map(|((name, &linear), &pixel_type)| dwa::Channel {
                    name,
                    pixel_type: DwaPixelType::from_code(pixel_type.0).expect("a pixel type"),
                    linear,
                })
                .collect();
            let ac_compression = if dwa.deflate_ac {
                dwa::AcCompression::Deflate
            } else {
                dwa::AcCompression::Huffman
            };
            scratch
                .dwa
                .compress_into(
                    &scratch.buf,
                    &channels,
                    width_size,
                    num_lines as usize,
                    dwa.level,
                    ac_compression,
                    &mut scratch.tmp_buf,
                )
                .expect("a block of the channels' size");
            out_data.extend_from_slice(&scratch.tmp_buf);
        }
    } else {
        let _ = (zfp, dwa);
        return Err("");
    }

    Ok(())
}

/// What a block of DWAA or DWAB needs of the header: the channels' names and `p_linear` flags,
/// the level, and whether its AC coefficients are compressed by ZIP (for tiled DWAA) or
/// Huffman coding otherwise.
#[allow(dead_code)]
pub(super) struct DwaParam {
    names: Vec<String>,
    linear: Vec<bool>,
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
        let linear = exr_header
            .channels
            .iter()
            .map(|channel| channel.p_linear != 0)
            .collect();
        DwaParam {
            names,
            linear,
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
#[must_use]
pub fn dwa_compression_level_attribute(level: f32) -> Attribute {
    Attribute {
        name: "dwaCompressionLevel".to_string(),
        type_name: "float".to_string(),
        value: level.to_le_bytes().to_vec(),
    }
}

/// What the port refuses to write with DWAA or DWAB compression, as OpenEXR refuses it.
pub const DWA_INVALID_LEVEL: &str = "Invalid dwa quality level specified";

/// Encodes one tile at `tile_idx` (`0..num_tiles`) into a complete 20-byte-prefixed chunk, or
/// returns an error reason (`None` if the encoded payload came out empty or overflowed `i32`).
#[allow(clippy::too_many_arguments)]
fn encode_one_tile_chunk(
    tile_idx: i32,
    num_x_tiles: i32,
    level_image: &ExrImage,
    exr_header: &ExrHeader,
    channel_offset_list: &[usize],
    pixel_data_size: usize,
    zfp: &ZfpParam,
    dwa: &DwaParam,
    as_tinyexr: bool,
    scratch: &mut EncodeScratch,
) -> Result<Vec<u8>, &'static str> {
    let x_tile = tile_idx % num_x_tiles;
    let y_tile = tile_idx / num_x_tiles;
    let tile = &level_image.tiles[tile_idx as usize];

    let mut data = vec![0u8; 5 * 4];
    let data_header_size = data.len();
    encode_pixel_data(
        &mut data,
        &tile.images,
        exr_header.compression_type,
        tile.width,
        exr_header.tile_size_x,
        0,
        tile.height,
        pixel_data_size,
        &exr_header.requested_pixel_types,
        channel_offset_list,
        zfp,
        dwa,
        as_tinyexr,
        scratch,
    )?;
    if data.len() <= data_header_size {
        return Err("");
    }
    let Ok(data_len) = i32::try_from(data.len() - data_header_size) else {
        return Err("");
    };
    data[0..4].copy_from_slice(&x_tile.to_le_bytes());
    data[4..8].copy_from_slice(&y_tile.to_le_bytes());
    data[8..12].copy_from_slice(&level_image.level_x.to_le_bytes());
    data[12..16].copy_from_slice(&level_image.level_y.to_le_bytes());
    data[16..20].copy_from_slice(&data_len.to_le_bytes());
    Ok(data)
}

/// `EncodeTiledLevel`: the tiles of one level, each as a block in `data_list` from
/// `start_index` on.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    as_tinyexr: bool,
    threads: usize,
    scratch: &mut EncodeScratch,
) -> Result<(), ExrError> {
    let Some(num_tiles) = num_x_tiles.checked_mul(num_y_tiles).filter(|&n| n >= 0) else {
        return Err(ExrError::new(
            ErrorCode::INVALID_ARGUMENT,
            "Invalid number of tiles in argument.\n",
        ));
    };
    if i64::from(num_tiles) != level_image.tiles.len() as i64 {
        return Err(ExrError::new(
            ErrorCode::INVALID_ARGUMENT,
            "Invalid number of tiles in argument.\n",
        ));
    }

    if (exr_header.tile_size_x > level_image.width || exr_header.tile_size_y > level_image.height)
        && level_image.level_x == 0
        && level_image.level_y == 0
    {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Failed to encode tile data.\n",
        ));
    }

    let total = num_tiles as usize;
    let level_slice = &mut data_list[start_index..start_index + total];

    #[cfg(feature = "std")]
    if threads > 1 && total > 1 {
        let workers = threads.min(total);
        let chunk_len = total.div_ceil(workers);
        let mut errors: Vec<Option<&'static str>> = vec![None; total];

        std::thread::scope(|s| {
            let mut d_chunks: Vec<&mut [Vec<u8>]> = level_slice.chunks_mut(chunk_len).collect();
            let mut e_chunks: Vec<&mut [Option<&'static str>]> =
                errors.chunks_mut(chunk_len).collect();
            let actual_workers = d_chunks.len();
            let last_d = d_chunks.pop().expect("at least one chunk");
            let last_e = e_chunks.pop().expect("at least one chunk");

            for (w_idx, (d_part, e_part)) in d_chunks.into_iter().zip(e_chunks).enumerate() {
                let base = (w_idx * chunk_len) as i32;
                s.spawn(move || {
                    let mut local_scratch = EncodeScratch::default();
                    for (k, (d_slot, e_slot)) in
                        d_part.iter_mut().zip(e_part.iter_mut()).enumerate()
                    {
                        match encode_one_tile_chunk(
                            base + k as i32,
                            num_x_tiles,
                            level_image,
                            exr_header,
                            channel_offset_list,
                            pixel_data_size,
                            zfp,
                            dwa,
                            as_tinyexr,
                            &mut local_scratch,
                        ) {
                            Ok(chunk) => *d_slot = chunk,
                            Err(reason) => *e_slot = Some(reason),
                        }
                    }
                });
            }

            let last_base = ((actual_workers - 1) * chunk_len) as i32;
            for (k, (d_slot, e_slot)) in last_d.iter_mut().zip(last_e.iter_mut()).enumerate() {
                match encode_one_tile_chunk(
                    last_base + k as i32,
                    num_x_tiles,
                    level_image,
                    exr_header,
                    channel_offset_list,
                    pixel_data_size,
                    zfp,
                    dwa,
                    as_tinyexr,
                    scratch,
                ) {
                    Ok(chunk) => *d_slot = chunk,
                    Err(reason) => *e_slot = Some(reason),
                }
            }
        });

        let mut failures = String::new();
        let mut invalid_data = false;
        for err in errors.into_iter().flatten() {
            failures.push_str(err);
            invalid_data = true;
        }
        if invalid_data {
            failures.push_str("Failed to encode tile data.\n");
            return Err(ExrError::new(ErrorCode::INVALID_DATA, failures));
        }
        return Ok(());
    }

    let _ = threads;
    let mut failures = String::new();
    let mut invalid_data = false;
    for (k, slot) in level_slice.iter_mut().enumerate() {
        match encode_one_tile_chunk(
            k as i32,
            num_x_tiles,
            level_image,
            exr_header,
            channel_offset_list,
            pixel_data_size,
            zfp,
            dwa,
            as_tinyexr,
            scratch,
        ) {
            Ok(chunk) => *slot = chunk,
            Err(reason) => {
                failures.push_str(reason);
                invalid_data = true;
            }
        }
    }

    if invalid_data {
        failures.push_str("Failed to encode tile data.\n");
        return Err(ExrError::new(ErrorCode::INVALID_DATA, failures));
    }
    Ok(())
}

/// Encodes one scanline block `i` into an 8-byte-prefixed chunk.
#[allow(clippy::too_many_arguments)]
fn encode_one_scanline_chunk(
    i: i32,
    num_scanlines: i32,
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
    channel_offset_list: &[usize],
    pixel_data_size: usize,
    zfp: &ZfpParam,
    dwa: &DwaParam,
    as_tinyexr: bool,
    scratch: &mut EncodeScratch,
) -> Result<Vec<u8>, &'static str> {
    let start_y = (i64::from(num_scanlines) * i64::from(i)) as i32;
    let end_y =
        (i64::from(num_scanlines) * (i64::from(i) + 1)).min(i64::from(exr_image.height)) as i32;
    let num_lines = end_y - start_y;

    let mut data = vec![0u8; 2 * 4];
    let data_header_size = data.len();

    encode_pixel_data(
        &mut data,
        &exr_image.images,
        exr_header.compression_type,
        exr_image.width,
        exr_image.width,
        start_y,
        num_lines,
        pixel_data_size,
        &exr_header.requested_pixel_types,
        channel_offset_list,
        zfp,
        dwa,
        as_tinyexr,
        scratch,
    )?;
    if data.len() <= data_header_size {
        return Err("");
    }
    let Ok(data_len) = i32::try_from(data.len() - data_header_size) else {
        return Err("");
    };
    data[0..4].copy_from_slice(&start_y.to_le_bytes());
    data[4..8].copy_from_slice(&data_len.to_le_bytes());
    Ok(data)
}

/// `EncodeChunk`: every block of the image, in `data_list`, and where each will be in the
/// file, in `offset_data`; with them, the size of the file.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn encode_chunk(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
    num_blocks: i32,
    chunk_offset: u64,
    is_multipart: bool,
    offset_data: &mut OffsetData,
    data_list: &mut Vec<Vec<u8>>,
    as_tinyexr: bool,
    threads: usize,
    scratch: &mut EncodeScratch,
) -> Result<u64, ExrError> {
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
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Invalid requested_pixel_type.\n",
                ));
            }
        }
    }

    let zfp = zfp_param_for_writing(&exr_header.custom_attributes);
    let dwa = DwaParam::new(exr_header, !exr_image.tiles.is_empty());

    let mut offset = chunk_offset;
    let doffset = if is_multipart { 4u64 } else { 0 };

    if exr_image.tiles.is_empty() {
        #[cfg(feature = "std")]
        let total = num_blocks as usize;

        #[cfg(feature = "std")]
        if threads > 1 && total > 1 {
            let workers = threads.min(total);
            let chunk_len = total.div_ceil(workers);
            let mut errors: Vec<Option<&'static str>> = vec![None; total];

            std::thread::scope(|s| {
                let mut d_chunks: Vec<&mut [Vec<u8>]> = data_list.chunks_mut(chunk_len).collect();
                let mut e_chunks: Vec<&mut [Option<&'static str>]> =
                    errors.chunks_mut(chunk_len).collect();
                let actual_workers = d_chunks.len();
                let last_d = d_chunks.pop().expect("at least one chunk");
                let last_e = e_chunks.pop().expect("at least one chunk");

                for (w_idx, (d_part, e_part)) in d_chunks.into_iter().zip(e_chunks).enumerate() {
                    let base = (w_idx * chunk_len) as i32;
                    let zfp_ref = &zfp;
                    let dwa_ref = &dwa;
                    let ch_offsets = &channel_offset_list;
                    s.spawn(move || {
                        let mut local_scratch = EncodeScratch::default();
                        for (k, (d_slot, e_slot)) in
                            d_part.iter_mut().zip(e_part.iter_mut()).enumerate()
                        {
                            match encode_one_scanline_chunk(
                                base + k as i32,
                                num_scanlines,
                                exr_image,
                                exr_header,
                                ch_offsets,
                                pixel_data_size,
                                zfp_ref,
                                dwa_ref,
                                as_tinyexr,
                                &mut local_scratch,
                            ) {
                                Ok(chunk) => *d_slot = chunk,
                                Err(reason) => *e_slot = Some(reason),
                            }
                        }
                    });
                }

                let last_base = ((actual_workers - 1) * chunk_len) as i32;
                for (k, (d_slot, e_slot)) in last_d.iter_mut().zip(last_e.iter_mut()).enumerate() {
                    match encode_one_scanline_chunk(
                        last_base + k as i32,
                        num_scanlines,
                        exr_image,
                        exr_header,
                        &channel_offset_list,
                        pixel_data_size,
                        &zfp,
                        &dwa,
                        as_tinyexr,
                        scratch,
                    ) {
                        Ok(chunk) => *d_slot = chunk,
                        Err(reason) => *e_slot = Some(reason),
                    }
                }
            });

            let mut failures = String::new();
            let mut invalid_data = false;
            for err in errors.into_iter().flatten() {
                failures.push_str(err);
                invalid_data = true;
            }
            if invalid_data {
                failures.push_str("Failed to encode scanline data.\n");
                return Err(ExrError::new(ErrorCode::INVALID_DATA, failures));
            }

            let offsets = &mut offset_data.offsets[0][0];
            for (slot, data) in offsets.iter_mut().zip(data_list.iter()) {
                *slot = offset;
                offset += data.len() as u64 + doffset;
            }
            return Ok(offset);
        }

        let mut failures = String::new();
        let mut invalid_data = false;

        for (i, slot) in data_list.iter_mut().enumerate() {
            match encode_one_scanline_chunk(
                i as i32,
                num_scanlines,
                exr_image,
                exr_header,
                &channel_offset_list,
                pixel_data_size,
                &zfp,
                &dwa,
                as_tinyexr,
                scratch,
            ) {
                Ok(chunk) => *slot = chunk,
                Err(reason) => {
                    failures.push_str(reason);
                    invalid_data = true;
                }
            }
        }

        if invalid_data {
            failures.push_str("Failed to encode scanline data.\n");
            return Err(ExrError::new(ErrorCode::INVALID_DATA, failures));
        }

        let offsets = &mut offset_data.offsets[0][0];
        for (slot, data) in offsets.iter_mut().zip(data_list.iter()) {
            *slot = offset;
            offset += data.len() as u64 + doffset;
        }

        Ok(offset)
    } else {
        let mut level_image = Some(exr_image);
        let mut block_idx = 0usize;
        let num_levels = if exr_header.tile_level_mode == TileLevelMode::RIPMAP_LEVELS {
            offset_data
                .num_x_levels
                .wrapping_mul(offset_data.num_y_levels)
        } else {
            offset_data.num_x_levels
        };
        for level in 0..num_levels {
            let Some(image) = level_image else {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Invalid number of tiled levels for EncodeChunk\n",
                ));
            };

            let level_index_from_image = level_index(
                image.level_x,
                image.level_y,
                exr_header.tile_level_mode,
                offset_data.num_x_levels,
            );
            if level_index_from_image < 0 {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Invalid tile level mode\n",
                ));
            }

            if level_index_from_image != level {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Incorrect level ordering in tiled image\n",
                ));
            }
            let num_y_tiles = offset_data.offsets[level as usize].len() as i32;
            if num_y_tiles <= 0 {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Invalid Y tile size\n",
                ));
            }

            let num_x_tiles = offset_data.offsets[level as usize][0].len() as i32;
            if num_x_tiles <= 0 {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Invalid X tile size\n",
                ));
            }

            encode_tiled_level(
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
                as_tinyexr,
                threads,
                scratch,
            )?;

            for row in &mut offset_data.offsets[level as usize][..num_y_tiles as usize] {
                for slot in &mut row[..num_x_tiles as usize] {
                    *slot = offset;
                    offset += data_list[block_idx].len() as u64 + doffset;
                    block_idx += 1;
                }
            }
            level_image = image.next_level.as_deref();
        }
        if block_idx as i64 != i64::from(num_blocks) {
            return Err(ErrorCode::INVALID_DATA.into());
        }
        Ok(offset)
    }
}

/// Validates caller-supplied `ExrImage` and `ExrHeader` before encoding.
#[allow(clippy::too_many_lines)]
fn check(exr_image: &ExrImage, exr_header: &ExrHeader, as_tinyexr: bool) -> Result<(), ExrError> {
    let refuse = |message: &str| {
        Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: message.to_string(),
        })
    };
    if !as_tinyexr
        && (exr_header.compression_type == Compression::DWAA
            || exr_header.compression_type == Compression::DWAB)
    {
        #[cfg(feature = "dwa")]
        {
            let level = dwa_level(&exr_header.custom_attributes);
            if !(0.0..=65504.0 * 100_000.0).contains(&level) {
                return refuse(DWA_INVALID_LEVEL);
            }
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

    let check_images = |images: &[ChannelData], needed: usize| {
        if images.len() != num_channels {
            return refuse("An image or a tile needs the values of each channel.");
        }
        for (c, image) in images.iter().enumerate() {
            let (pixel_type, len) = (image.pixel_type(), image.len());
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

    #[cfg(feature = "zfp")]
    let zfp = (exr_header.compression_type == Compression::ZFP)
        .then(|| zfp_param_for_writing(&exr_header.custom_attributes));
    let check_zfp = |sizes: &[(i32, i32)]| {
        #[cfg(not(feature = "zfp"))]
        {
            let _ = sizes;
            if exr_header.compression_type == Compression::ZFP {
                return Err(ExrError {
                    code: ErrorCode::UNSUPPORTED_FEATURE,
                    message: "ZFP compression requires the `zfp` Cargo feature.".to_string(),
                });
            }
            Ok(())
        }
        #[cfg(feature = "zfp")]
        {
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
        }
    };

    if exr_image.tiles.is_empty() {
        if exr_image.width < 1 || exr_image.height < 1 {
            return refuse("An image needs a width and a height.");
        }
        let Some(needed) = (exr_image.width as usize).checked_mul(exr_image.height as usize) else {
            return refuse("A channel has fewer values than the image or the tile has pixels.");
        };
        check_images(&exr_image.images, needed)?;
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
    for image in exr_image.levels() {
        for tile in &image.tiles {
            let ExrTile { width, height, .. } = *tile;
            if width < 1 || height < 1 {
                return refuse("A tile needs a width and a height.");
            }
            let Some(needed) = (height as usize - 1)
                .checked_mul(exr_header.tile_size_x as usize)
                .and_then(|n| n.checked_add(width as usize))
            else {
                return refuse("A channel has fewer values than the image or the tile has pixels.");
            };
            check_images(&tile.images, needed)?;
        }
    }
    let mut sizes = Vec::new();
    for image in exr_image.levels() {
        sizes.extend(image.tiles.iter().map(|tile| (tile.width, tile.height)));
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
/// integers as integers); `compression_type` is one of the `COMPRESSIONTYPE_*`, and
/// `custom_attributes` are written after the attributes every file has.
///
/// [`tile_levels`]: super::tile_levels
///
/// # Errors
///
/// A header or image tinyexr would refuse to write, with tinyexr's error code and message.
pub fn save_image(exr_image: &ExrImage, exr_header: &ExrHeader) -> Result<Vec<u8>, ExrError> {
    save_parts(&[(exr_image, exr_header)], false, 1)
}

/// Multi-threaded [`save_image`] compressing scanline blocks or tiles in parallel across
/// `num_threads` worker threads (`0` selects [`std::thread::available_parallelism`], `1` runs
/// on the calling thread).
///
/// Produces a **byte-for-byte identical** `.exr` file to [`save_image`] regardless of thread
/// count.
///
/// # Errors
///
/// As [`save_image`].
#[cfg(feature = "std")]
pub fn save_image_parallel(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
    num_threads: usize,
) -> Result<Vec<u8>, ExrError> {
    save_parts(&[(exr_image, exr_header)], false, num_threads)
}

/// `SaveEXRImageToMemory` exactly as tinyexr has it: the same file, byte for byte, for every
/// image.
///
/// [`load_image_as_tinyexr`]: super::load_image_as_tinyexr
///
/// # Errors
///
/// As [`save_image`].
pub fn save_image_as_tinyexr(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
) -> Result<Vec<u8>, ExrError> {
    save_parts(&[(exr_image, exr_header)], true, 1)
}

/// `SaveEXRMultipartImageToMemory`: images as the parts of one `.exr` file, each with its
/// header, as [`save_image`] writes one.
///
/// # Errors
///
/// No parts, a part without a name of its own, or a part [`save_image`] would refuse, with
/// tinyexr's error code and message.
pub fn save_multipart_images(parts: &[(&ExrImage, &ExrHeader)]) -> Result<Vec<u8>, ExrError> {
    save_multipart(parts, false)
}

/// `SaveEXRMultipartImageToMemory` exactly as tinyexr has it
/// ([`save_image_as_tinyexr`] says where that differs).
///
/// # Errors
///
/// As [`save_multipart_images`].
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
    save_parts(parts, as_tinyexr, 1)
}

/// `SaveEXRNPartImageToMemory`.
#[allow(clippy::too_many_lines)]
fn save_parts(
    parts: &[(&ExrImage, &ExrHeader)],
    as_tinyexr: bool,
    num_threads: usize,
) -> Result<Vec<u8>, ExrError> {
    let fail = |code: ErrorCode, message: &str| {
        Err(ExrError {
            code,
            message: message.to_string(),
        })
    };
    let num_parts = parts.len();
    let threads = effective_threads(num_threads);

    for &(_, exr_header) in parts {
        if exr_header.compression_type.0 < 0 {
            return fail(
                ErrorCode::SERIALIZATION_FAILED,
                "Invalid argument for SaveEXRNPartImageToMemory",
            );
        }
        if exr_header.compression_type == Compression::ZFP
            && exr_header
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
    for &(exr_image, exr_header) in parts {
        check(exr_image, exr_header, as_tinyexr)?;
    }

    let mut memory = Vec::new();

    // Header
    memory.extend_from_slice(&[0x76, 0x2f, 0x31, 0x01]);

    // Version
    let long_name = parts[0].1.long_name;
    {
        let mut marker = [2u8, 0, 0, 0];
        if num_parts == 1 && !parts[0].0.tiles.is_empty() {
            marker[1] |= 0x2;
        }
        if long_name {
            marker[1] |= 0x4;
        }
        if num_parts > 1 {
            marker[1] |= 0x10;
        }
        memory.extend_from_slice(&marker);
    }

    let mut total_chunk_count = 0u64;
    let mut chunk_count = vec![0i32; num_parts];
    let mut offset_data: Vec<OffsetData> = (0..num_parts).map(|_| OffsetData::default()).collect();
    for (i, &(exr_image, exr_header)) in parts.iter().enumerate() {
        if exr_image.tiles.is_empty() {
            let num_scanlines = num_scanline_blocks(exr_header.compression_type, as_tinyexr);
            let nblocks = (i64::from(exr_image.height) + i64::from(num_scanlines) - 1)
                / i64::from(num_scanlines);
            let Ok(nblocks_i32) = i32::try_from(nblocks) else {
                return fail(ErrorCode::DATA_TOO_LARGE, "");
            };
            chunk_count[i] = nblocks_i32;
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
        total_chunk_count += chunk_count[i] as u64;
    }

    // Write attributes to memory buffer.
    let mut partnames: Vec<Vec<u8>> = Vec::new();
    for (i, &(exr_image, exr_header)) in parts.iter().enumerate() {
        {
            let data = write_channel_info(
                &exr_header.channels,
                &exr_header.requested_pixel_types,
                as_tinyexr,
            );
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
            write_attribute_to_memory(&mut memory, "displayWindow", "box2i", &data0);
        }

        write_attribute_to_memory(&mut memory, "lineOrder", "lineOrder", &[0]);
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

        if num_parts > 1 {
            {
                let name = exr_header.name.as_bytes();
                let name = &name[..name
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(name.len())
                    .min(255)];
                if name.is_empty() {
                    return fail(
                        ErrorCode::SERIALIZATION_FAILED,
                        "Invalid 'name' attribute for a multi-part file",
                    );
                }
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
            }
            {
                let type_name = if exr_image.tiles.is_empty() {
                    "scanlineimage"
                } else {
                    "tiledimage"
                };
                write_attribute_to_memory(&mut memory, "type", "string", type_name.as_bytes());
            }
            write_attribute_to_memory(
                &mut memory,
                "chunkCount",
                "int",
                &chunk_count[i].to_le_bytes(),
            );
        }

        for attribute in &exr_header.custom_attributes {
            write_attribute_to_memory(
                &mut memory,
                &attribute.name,
                &attribute.type_name,
                &attribute.value,
            );
        }

        memory.push(0);
    }
    if num_parts > 1 {
        memory.push(0);
    }

    let mut chunk_offset = memory.len() as u64 + total_chunk_count * 8;

    let mut total_size = 0u64;
    let mut data_lists: Vec<Vec<Vec<u8>>> = vec![Vec::new(); num_parts];
    let mut scratch = EncodeScratch::default();
    for (i, &(exr_image, exr_header)) in parts.iter().enumerate() {
        total_size = encode_chunk(
            exr_image,
            exr_header,
            chunk_count[i],
            chunk_offset,
            num_parts > 1,
            &mut offset_data[i],
            &mut data_lists[i],
            as_tinyexr,
            threads,
            &mut scratch,
        )
        .map_err(|error| ExrError::new(ErrorCode::SERIALIZATION_FAILED, error.message))?;
        chunk_offset = total_size;
    }

    if total_size == 0 {
        return fail(ErrorCode::INVALID_DATA, "Output memory size is zero");
    }

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
///
/// # Errors
///
/// An unsupported number of components, or fewer floats than the size asks for, with tinyexr's
/// error code and message.
pub fn save_floats(
    data: &[f32],
    width: i32,
    height: i32,
    components: i32,
    save_as_fp16: bool,
) -> Result<Vec<u8>, ExrError> {
    save_floats_with_threads(data, width, height, components, save_as_fp16, 1)
}

/// Multi-threaded [`save_floats`] compressing ZIP blocks in parallel across `num_threads`
/// worker threads (`0` selects [`std::thread::available_parallelism`], `1` runs on the
/// calling thread).
///
/// Produces a **byte-for-byte identical** `.exr` file to [`save_floats`] regardless of thread
/// count.
///
/// # Errors
///
/// As [`save_floats`].
#[cfg(feature = "std")]
pub fn save_floats_parallel(
    data: &[f32],
    width: i32,
    height: i32,
    components: i32,
    save_as_fp16: bool,
    num_threads: usize,
) -> Result<Vec<u8>, ExrError> {
    save_floats_with_threads(data, width, height, components, save_as_fp16, num_threads)
}

fn save_floats_with_threads(
    data: &[f32],
    width: i32,
    height: i32,
    components: i32,
    save_as_fp16: bool,
    num_threads: usize,
) -> Result<Vec<u8>, ExrError> {
    if !(components == 1 || components == 3 || components == 4) {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: format!("Unsupported component value : {components}\n"),
        });
    }
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

    let Some(pixels) = (width as usize).checked_mul(height as usize) else {
        return Err(ExrError {
            code: ErrorCode::DATA_TOO_LARGE,
            message: String::new(),
        });
    };

    let mut header = ExrHeader {
        compression_type: if width < 16 && height < 16 {
            Compression::NONE
        } else {
            Compression::ZIP
        },
        ..ExrHeader::default()
    };

    let components_size = components as usize;
    let split: Vec<Vec<f32>> = (0..components_size)
        .map(|k| (0..pixels).map(|i| data[components_size * i + k]).collect())
        .collect();

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
    header.pixel_types = vec![PixelType::FLOAT; components_size];
    header.requested_pixel_types = vec![
        if save_as_fp16 {
            PixelType::HALF
        } else {
            PixelType::FLOAT
        };
        components_size
    ];

    let memory = save_parts(&[(&image, &header)], false, num_threads)?;
    if memory.len() > i32::MAX as usize {
        return Err(ExrError {
            code: ErrorCode::DATA_TOO_LARGE,
            message: String::new(),
        });
    }
    Ok(memory)
}
