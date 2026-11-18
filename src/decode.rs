//! A block of pixels out of a file and into the channels of an image: a port of
//! `DecodePixelData`, `DecodeTiledPixelData`, `ComputeChannelLayout` and `AllocateImage` of
//! tinyexr v3.4.0 (`tinyexr.h`).

use super::b44::decompress_b44;
use super::piz::decompress_piz;
use super::pxr24::decompress_pxr24;
use super::zfp::{ZfpParam, decompress_zfp};
use super::zip_rle::{decompress_rle, decompress_zip};
use super::{ChannelInfo, Compression, LineOrder, PixelType, half_to_float};
use openexr_nrp::{PixelType as DwaPixelType, dwa};

/// The values of one channel of an image or of a tile, line after line. (The original has a
/// block of bytes for each channel, to be read as the type that was asked for.)
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelData {
    /// 32-bit unsigned integers.
    Uint(Vec<u32>),
    /// Halves, as their bits.
    Half(Vec<u16>),
    /// Floats.
    Float(Vec<f32>),
}

/// `ComputeChannelLayout`: where each channel's values start within a pixel's bytes in a
/// file, and how many bytes a pixel takes. `None` for a channel of an unknown type.
pub(crate) fn compute_channel_layout(channels: &[ChannelInfo]) -> Option<(Vec<usize>, i32)> {
    let mut channel_offset_list = Vec::with_capacity(channels.len());
    let mut pixel_data_size = 0i32;
    let mut channel_offset = 0usize;

    for channel in channels {
        channel_offset_list.push(channel_offset);
        if channel.pixel_type == PixelType::HALF {
            pixel_data_size += 2;
            channel_offset += 2;
        } else if channel.pixel_type == PixelType::FLOAT || channel.pixel_type == PixelType::UINT {
            pixel_data_size += 4;
            channel_offset += 4;
        } else {
            // ???
            return None;
        }
    }
    Some((channel_offset_list, pixel_data_size))
}

/// `AllocateImage`: room for `data_width` by `data_height` values of each channel, as the
/// type that was asked for: a channel of halves as halves or as floats, the others as they
/// are. `None` where a channel's type or the type asked for is not one of those, or where
/// there is not the memory for it. (The original's memory starts as whatever it holds; here
/// it starts as zeros.)
pub(crate) fn allocate_image(
    channels: &[ChannelInfo],
    requested_pixel_types: &[PixelType],
    data_width: i32,
    data_height: i32,
) -> Option<Vec<ChannelData>> {
    let mut images = Vec::with_capacity(channels.len());
    // (Sizes below zero are sizes too large to have, in the original's unsigned arithmetic.)
    let data_len = (data_width.max(0) as usize).checked_mul(data_height.max(0) as usize)?;
    if data_width < 0 || data_height < 0 {
        return None;
    }

    for (c, channel) in channels.iter().enumerate() {
        if channel.pixel_type == PixelType::HALF {
            // pixel_data_size += sizeof(unsigned short);
            // channel_offset += sizeof(unsigned short);
            // Alloc internal image for half type.
            if requested_pixel_types[c] == PixelType::HALF {
                images.push(ChannelData::Half(zeros(data_len)?));
            } else if requested_pixel_types[c] == PixelType::FLOAT {
                images.push(ChannelData::Float(zeros(data_len)?));
            } else {
                return None; // Unsupported
            }
        } else if channel.pixel_type == PixelType::FLOAT {
            images.push(ChannelData::Float(zeros(data_len)?));
        } else if channel.pixel_type == PixelType::UINT {
            images.push(ChannelData::Uint(zeros(data_len)?));
        } else {
            return None; // Unsupported
        }
    }

    Some(images)
}

/// `len` zeros, or `None` where there is not the memory for them: a damaged file can ask for
/// more than there is, which the original answers with a message too.
fn zeros<T: Clone + Default>(len: usize) -> Option<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, T::default());
    Some(values)
}

/// Where the line `v` of a block and its value `u` go in a channel's values: the original's
/// `(line_no + v) * x_stride + u`, or the same counted from the bottom line. (All in the
/// original's unsigned arithmetic, so that what would be before the start is far past the
/// end.)
#[inline]
fn image_index(
    line_order: LineOrder,
    height: i32,
    line_no: i32,
    x_stride: i32,
    v: usize,
    u: usize,
) -> usize {
    let line = if line_order == LineOrder::INCREASING_Y {
        (line_no as isize as usize).wrapping_add(v)
    } else {
        (height as isize as usize)
            .wrapping_sub(1)
            .wrapping_sub((line_no as isize as usize).wrapping_add(v))
    };
    line.wrapping_mul(x_stride as isize as usize)
        .wrapping_add(u)
}

/// The halves of `src`, two bytes each, into consecutive places of a channel from `start`:
/// as many as fit, and whether all of them did. (Where one does not, the per-value loop of
/// the original stops, having written those before it.)
fn put_halves(image: &mut ChannelData, start: usize, src: &[u8]) -> bool {
    let (src, _) = src.as_chunks::<2>();
    let fits = |len: usize| len.checked_sub(start).is_some_and(|room| room >= src.len());
    match image {
        ChannelData::Half(values) => {
            let all = fits(values.len());
            for (value, bytes) in values
                .get_mut(start..)
                .unwrap_or_default()
                .iter_mut()
                .zip(src)
            {
                *value = u16::from_le_bytes(*bytes);
            }
            all
        }
        ChannelData::Float(values) => {
            let all = fits(values.len());
            for (value, bytes) in values
                .get_mut(start..)
                .unwrap_or_default()
                .iter_mut()
                .zip(src)
            {
                *value = half_to_float(u16::from_le_bytes(*bytes));
            }
            all
        }
        ChannelData::Uint(values) => {
            let all = fits(values.len());
            for (value, bytes) in values
                .get_mut(start..)
                .unwrap_or_default()
                .iter_mut()
                .zip(src)
            {
                *value = half_to_float(u16::from_le_bytes(*bytes)).to_bits();
            }
            all
        }
    }
}

/// The four-byte values of `src` into consecutive places of a channel of floats or of
/// integers from `start`, as [`put_halves`] puts halves.
fn put_fours(image: &mut ChannelData, start: usize, src: &[u8]) -> bool {
    let (src, _) = src.as_chunks::<4>();
    let fits = |len: usize| len.checked_sub(start).is_some_and(|room| room >= src.len());
    match image {
        ChannelData::Uint(values) => {
            let all = fits(values.len());
            for (value, bytes) in values
                .get_mut(start..)
                .unwrap_or_default()
                .iter_mut()
                .zip(src)
            {
                *value = u32::from_le_bytes(*bytes);
            }
            all
        }
        ChannelData::Float(values) => {
            let all = fits(values.len());
            for (value, bytes) in values
                .get_mut(start..)
                .unwrap_or_default()
                .iter_mut()
                .zip(src)
            {
                *value = f32::from_le_bytes(*bytes);
            }
            all
        }
        ChannelData::Half(_) => src.is_empty(),
    }
}

/// What the PIZ, ZIP, RLE and PXR24 blocks of the original each end with: the lines of a
/// decompressed block, in which each line holds its channels one after the other, into the
/// channels of the image.
// (Its parameters are tinyexr's, each a separate part of the state.)
#[allow(clippy::too_many_arguments)]
fn unpack_lines(
    out_images: &mut [ChannelData],
    requested_pixel_types: &[PixelType],
    out_buf: &[u8],
    line_order: LineOrder,
    width: i32,
    height: i32,
    x_stride: i32,
    line_no: i32,
    num_lines: i32,
    pixel_data_size: usize,
    channels: &[ChannelInfo],
    channel_offset_list: &[usize],
) -> bool {
    let width = width as usize;
    for (c, channel) in channels.iter().enumerate() {
        let size = if channel.pixel_type == PixelType::HALF {
            2
        } else if channel.pixel_type == PixelType::UINT || channel.pixel_type == PixelType::FLOAT {
            if requested_pixel_types[c] != channel.pixel_type {
                return false;
            }
            4
        } else {
            return false;
        };
        for v in 0..num_lines as usize {
            let line_ptr = v * pixel_data_size * width + channel_offset_list[c] * width;
            let line = &out_buf[line_ptr..line_ptr + size * width];
            let start = image_index(line_order, height, line_no, x_stride, v, 0);
            let stored = if size == 2 {
                put_halves(&mut out_images[c], start, line)
            } else {
                put_fours(&mut out_images[c], start, line)
            };
            if !stored {
                // (A place outside the image: the original writes there all the same.)
                return false;
            }
        }
    }
    true
}

/// The size of a decompressed block of `width` by `num_lines` pixels of `pixel_data_size`
/// bytes, or `None` where that is not a size (the original then asks for more memory than
/// there is).
fn block_size(width: i32, num_lines: i32, pixel_data_size: usize) -> Option<usize> {
    if width < 0 || num_lines < 0 {
        return None;
    }
    (width as usize)
        .checked_mul(num_lines as usize)?
        .checked_mul(pixel_data_size)
        .filter(|&size| size <= isize::MAX as usize / 2)
}

/// `DecodePixelData`: the block of `data_len` bytes at the start of `data` (which goes on
/// to the end of the file), decompressed and stored in `out_images`: `num_lines` lines of
/// `width` pixels, from the line `line_no` (or `y`, for a block that is not compressed) of
/// an image of `height` lines whose lines are `x_stride` values apart. `zfp` is what the
/// header's attributes say of ZFP compression, for a block of it.
// (Its parameters are tinyexr's, each a separate part of the state; one pass, step for step as
// tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn decode_pixel_data(
    out_images: &mut [ChannelData],
    requested_pixel_types: &[PixelType],
    data: &[u8],
    data_len: usize,
    compression_type: Compression,
    line_order: LineOrder,
    width: i32,
    height: i32,
    x_stride: i32,
    y: i32,
    line_no: i32,
    num_lines: i32,
    pixel_data_size: usize,
    channels: &[ChannelInfo],
    channel_offset_list: &[usize],
    zfp: Option<&ZfpParam>,
    as_tinyexr: bool,
) -> bool {
    let block = &data[..data_len];
    if compression_type == Compression::PIZ {
        // PIZ
        if width == 0 || num_lines == 0 || pixel_data_size == 0 {
            // Invalid input #90
            return false;
        }

        // Allocate original data size.
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
        let mut out_buf = vec![0u8; size];
        let channel_sizes: Vec<usize> = channels
            .iter()
            .map(|channel| {
                if channel.pixel_type == PixelType::HALF {
                    1
                } else {
                    2
                }
            })
            .collect();
        if !decompress_piz(
            &mut out_buf,
            data,
            size,
            data_len,
            &channel_sizes,
            width,
            num_lines,
        ) {
            return false;
        }

        // For PIZ_COMPRESSION:
        //   pixel sample data for channel 0 for scanline 0
        //   pixel sample data for channel 1 for scanline 0
        //   pixel sample data for channel ... for scanline 0
        //   pixel sample data for channel n for scanline 0
        //   pixel sample data for channel 0 for scanline 1
        //   pixel sample data for channel 1 for scanline 1
        //   pixel sample data for channel ... for scanline 1
        //   pixel sample data for channel n for scanline 1
        //   ...
        unpack_lines(
            out_images,
            requested_pixel_types,
            &out_buf,
            line_order,
            width,
            height,
            x_stride,
            line_no,
            num_lines,
            pixel_data_size,
            channels,
            channel_offset_list,
        )
    } else if compression_type == Compression::ZIPS || compression_type == Compression::ZIP {
        // Allocate original data size.
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
        let mut out_buf = vec![0u8; size];

        if size == 0 || decompress_zip(&mut out_buf, block).is_none() {
            return false;
        }

        // For ZIP_COMPRESSION: the same layout as PIZ.
        unpack_lines(
            out_images,
            requested_pixel_types,
            &out_buf,
            line_order,
            width,
            height,
            x_stride,
            line_no,
            num_lines,
            pixel_data_size,
            channels,
            channel_offset_list,
        )
    } else if compression_type == Compression::ZFP {
        // (`decode_chunk` has found the settings; this code path should not be reachable.)
        let Some(zfp) = zfp else { return false };

        // Allocate original data size.
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
        let mut out_buf = vec![0u8; size];
        if size == 0 {
            return false;
        }
        // (The answer is not looked at; but settings that ask for more memory than there is
        // stop the original.)
        if decompress_zfp(&mut out_buf, width, num_lines, channels.len(), block, zfp).is_none() {
            return false;
        }

        // For ZFP_COMPRESSION: the same layout as PIZ, but every channel has to be of floats.
        let width = width as usize;
        for (c, channel) in channels.iter().enumerate() {
            if channel.pixel_type != PixelType::FLOAT
                || requested_pixel_types[c] != PixelType::FLOAT
            {
                return false;
            }
            for v in 0..num_lines as usize {
                let line_ptr = v * pixel_data_size * width + channel_offset_list[c] * width;
                let start = image_index(line_order, height, line_no, x_stride, v, 0);
                if !put_fours(
                    &mut out_images[c],
                    start,
                    &out_buf[line_ptr..line_ptr + 4 * width],
                ) {
                    // (A place outside the image: the original writes there all the same.)
                    return false;
                }
            }
        }
        true
    } else if compression_type == Compression::RLE {
        // Allocate original data size.
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
        let mut out_buf = vec![0u8; size];

        if size == 0 {
            return false;
        }

        if !decompress_rle(&mut out_buf, size, block) {
            return false;
        }

        // For RLE_COMPRESSION: the same layout as PIZ.
        unpack_lines(
            out_images,
            requested_pixel_types,
            &out_buf,
            line_order,
            width,
            height,
            x_stride,
            line_no,
            num_lines,
            pixel_data_size,
            channels,
            channel_offset_list,
        )
    } else if compression_type == Compression::PXR24 {
        // PXR24 compression.
        // Allocate original data size.
        // (A block of no bytes stops the original, with an exception out of a C function.)
        let Some(size) = block_size(width, num_lines, pixel_data_size).filter(|&size| size != 0)
        else {
            return false;
        };
        // (The format's rule, which the original does not have for this compression: a block
        // as long as its pixels is its pixels.)
        if !as_tinyexr && data_len == size {
            return unpack_lines(
                out_images,
                requested_pixel_types,
                block,
                line_order,
                width,
                height,
                x_stride,
                line_no,
                num_lines,
                pixel_data_size,
                channels,
                channel_offset_list,
            );
        }
        let mut out_buf = vec![0u8; size];

        let pixel_types: Vec<PixelType> =
            channels.iter().map(|channel| channel.pixel_type).collect();
        if !decompress_pxr24(
            &mut out_buf,
            block,
            width,
            num_lines,
            &pixel_types,
            as_tinyexr,
        ) {
            return false;
        }

        // PXR24 outputs in the same layout as other compression methods.
        unpack_lines(
            out_images,
            requested_pixel_types,
            &out_buf,
            line_order,
            width,
            height,
            x_stride,
            line_no,
            num_lines,
            pixel_data_size,
            channels,
            channel_offset_list,
        )
    } else if compression_type == Compression::B44 || compression_type == Compression::B44A {
        // B44/B44A compression.
        // B44 output is always HALF (2 bytes per channel per pixel).
        // Calculate B44 output size (HALF channels stay as HALF, others as-is).
        let sampling = |value: i32| if value > 0 { value } else { 1 };
        let mut b44_out_size = 0usize;
        for channel in channels {
            let xs = sampling(channel.x_sampling);
            let ys = sampling(channel.y_sampling);
            let cw = (width.wrapping_add(xs).wrapping_sub(1) / xs) as isize as usize;
            let ch = (num_lines.wrapping_add(ys).wrapping_sub(1) / ys) as isize as usize;
            let bpp = if channel.pixel_type == PixelType::HALF {
                2usize
            } else {
                4
            };
            b44_out_size = b44_out_size.wrapping_add(cw.wrapping_mul(ch).wrapping_mul(bpp));
        }

        // (A block of no bytes stops the original, as does one too large to have.)
        if b44_out_size == 0 || b44_out_size > isize::MAX as usize / 2 {
            return false;
        }
        // (The format's rule, which the original does not have for this compression: a block
        // as long as its pixels is its pixels, line after line.)
        if !as_tinyexr && block_size(width, num_lines, pixel_data_size) == Some(data_len) {
            return unpack_lines(
                out_images,
                requested_pixel_types,
                block,
                line_order,
                width,
                height,
                x_stride,
                line_no,
                num_lines,
                pixel_data_size,
                channels,
                channel_offset_list,
            );
        }
        let mut out_buf = vec![0u8; b44_out_size];

        if !decompress_b44(&mut out_buf, block, width, num_lines, channels) {
            return false;
        }

        // B44 outputs channel by channel (not scanline interleaved), respecting per-channel
        // subsampling.
        let mut ch_offset = 0usize;
        for (c, channel) in channels.iter().enumerate() {
            let xs = sampling(channel.x_sampling);
            let ys = sampling(channel.y_sampling);
            let ch_width = (width.wrapping_add(xs).wrapping_sub(1) / xs) as usize;
            let ch_height = (num_lines.wrapping_add(ys).wrapping_sub(1) / ys) as usize;
            let ch_bytes = if channel.pixel_type == PixelType::HALF {
                2
            } else {
                4
            };

            if channel.pixel_type == PixelType::HALF {
                for v in 0..ch_height {
                    let line_ptr = ch_offset + v * ch_width * 2;
                    let start = image_index(line_order, height, line_no, x_stride, v, 0);
                    if !put_halves(
                        &mut out_images[c],
                        start,
                        &out_buf[line_ptr..line_ptr + 2 * ch_width],
                    ) {
                        return false;
                    }
                }
            } else if channel.pixel_type == PixelType::UINT
                || channel.pixel_type == PixelType::FLOAT
            {
                if requested_pixel_types[c] != channel.pixel_type {
                    return false;
                }
                for v in 0..ch_height {
                    let line_ptr = ch_offset + v * ch_width * 4;
                    let start = image_index(line_order, height, line_no, x_stride, v, 0);
                    if !put_fours(
                        &mut out_images[c],
                        start,
                        &out_buf[line_ptr..line_ptr + 4 * ch_width],
                    ) {
                        return false;
                    }
                }
            } else {
                return false;
            }

            ch_offset += ch_width * ch_height * ch_bytes;
        }
        true
    } else if !as_tinyexr
        && (compression_type == Compression::DWAA || compression_type == Compression::DWAB)
    {
        // (Not in the original.) A block as long as its pixels is its pixels.
        let Some(size) = block_size(width, num_lines, pixel_data_size).filter(|&size| size != 0)
        else {
            return false;
        };
        if data_len == size {
            return unpack_lines(
                out_images,
                requested_pixel_types,
                block,
                line_order,
                width,
                height,
                x_stride,
                line_no,
                num_lines,
                pixel_data_size,
                channels,
                channel_offset_list,
            );
        }
        let mut out_buf = vec![0u8; size];
        let names: Vec<&str> = channels
            .iter()
            .map(|channel| channel.name.split('\0').next().unwrap_or(""))
            .collect();
        let dwa_channels: Vec<dwa::Channel> = channels
            .iter()
            .zip(&names)
            .map(|(channel, name)| {
                Some(dwa::Channel {
                    name,
                    pixel_type: DwaPixelType::from_code(channel.pixel_type.0)?,
                    linear: channel.p_linear != 0,
                })
            })
            .collect::<Option<_>>()
            .unwrap_or_default();
        if dwa_channels.len() != channels.len() {
            return false;
        }
        if dwa::decompress(
            block,
            &dwa_channels,
            width as usize,
            num_lines as usize,
            &mut out_buf,
        )
        .is_err()
        {
            return false;
        }
        unpack_lines(
            out_images,
            requested_pixel_types,
            &out_buf,
            line_order,
            width,
            height,
            x_stride,
            line_no,
            num_lines,
            pixel_data_size,
            channels,
            channel_offset_list,
        )
    } else if compression_type == Compression::NONE {
        let width_size = width.max(0) as usize;
        for (c, channel) in channels.iter().enumerate() {
            for v in 0..num_lines.max(0) as usize {
                let line_ptr = v
                    .wrapping_mul(pixel_data_size)
                    .wrapping_mul(width as isize as usize)
                    .wrapping_add(channel_offset_list[c].wrapping_mul(width as isize as usize));
                // (Here the line is `y`, not `line_no`.)
                let out_line = image_index(line_order, height, y, x_stride, v, 0);

                if channel.pixel_type == PixelType::HALF {
                    if requested_pixel_types[c] == PixelType::HALF {
                        // (No look at whether the line is all within the block: the
                        // original reads on into what follows it in the file.)
                        // (Past the end of the file, which the original reads too, the line stops.)
                        let in_file = data.len().saturating_sub(line_ptr) / 2;
                        let line = &data[line_ptr.min(data.len())..][..2 * width_size.min(in_file)];
                        if !put_halves(&mut out_images[c], out_line, line) || in_file < width_size {
                            return false;
                        }
                    } else if requested_pixel_types[c] == PixelType::FLOAT {
                        if line_ptr
                            .checked_add(2 * width_size)
                            .is_none_or(|end| end > data_len)
                        {
                            // Insufficient data size.
                            return false;
                        }

                        let line = &data[line_ptr..line_ptr + 2 * width_size];
                        if !put_halves(&mut out_images[c], out_line, line) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else if channel.pixel_type == PixelType::FLOAT
                    || channel.pixel_type == PixelType::UINT
                {
                    if line_ptr
                        .checked_add(4 * width_size)
                        .is_none_or(|end| end > data_len)
                    {
                        // Insufficient data size.
                        return false;
                    }

                    let line = &data[line_ptr..line_ptr + 4 * width_size];
                    if !put_fours(&mut out_images[c], out_line, line) {
                        return false;
                    }
                }
            }
        }
        true
    } else {
        // (A kind of compression the original has no code for is taken for done.)
        true
    }
}

/// `DecodeTiledPixelData`: a tile's block into the tile's channels. The answer is the tile's
/// width and height, which are less than the tile size at the right and bottom edges.
// (Its parameters are tinyexr's, each a separate part of the state.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_tiled_pixel_data(
    out_images: &mut [ChannelData],
    requested_pixel_types: &[PixelType],
    data: &[u8],
    data_len: usize,
    compression_type: Compression,
    data_width: i32,
    data_height: i32,
    tile_offset_x: i32,
    tile_offset_y: i32,
    tile_size_x: i32,
    tile_size_y: i32,
    pixel_data_size: usize,
    channels: &[ChannelInfo],
    channel_offset_list: &[usize],
    zfp: Option<&ZfpParam>,
    as_tinyexr: bool,
) -> Option<(i32, i32)> {
    // Here, data_width and data_height are the dimensions of the current (sub)level.
    if tile_size_x.wrapping_mul(tile_offset_x) > data_width
        || tile_size_y.wrapping_mul(tile_offset_y) > data_height
    {
        return None;
    }

    // Compute actual image size in a tile.
    let width = if tile_offset_x.wrapping_add(1).wrapping_mul(tile_size_x) >= data_width {
        data_width.wrapping_sub(tile_offset_x.wrapping_mul(tile_size_x))
    } else {
        tile_size_x
    };
    let height = if tile_offset_y.wrapping_add(1).wrapping_mul(tile_size_y) >= data_height {
        data_height.wrapping_sub(tile_offset_y.wrapping_mul(tile_size_y))
    } else {
        tile_size_y
    };

    // Image size = tile size.
    decode_pixel_data(
        out_images,
        requested_pixel_types,
        data,
        data_len,
        compression_type,
        LineOrder::INCREASING_Y,
        width,
        tile_size_y,
        /* stride */ tile_size_x,
        /* y */ 0,
        /* line_no */ 0,
        height,
        pixel_data_size,
        channels,
        channel_offset_list,
        zfp,
        as_tinyexr,
    )
    .then_some((width, height))
}
