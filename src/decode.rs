//! A block of pixels out of a file and into the channels of an image: a port of
//! `DecodePixelData`, `DecodeTiledPixelData`, `ComputeChannelLayout` and `AllocateImage` of
//! tinyexr v3.4.0 (`tinyexr.h`).

use alloc::vec::Vec;

use super::b44::{decompress_b44, sampled_dim};
use super::piz::{PizScratch, decompress_piz_with_scratch};
use super::pxr24::decompress_pxr24_with_scratch;
use super::zfp::ZfpParam;
#[cfg(feature = "zfp")]
use super::zfp::decompress_zfp;
use super::zip_rle::{decompress_rle_with_scratch, decompress_zip_with_scratch};
use super::{ChannelInfo, Compression, LineOrder, PixelType, half_to_float};
#[cfg(feature = "dwa")]
use openexr_nrp::{PixelType as DwaPixelType, dwa};

/// Reusable scratch buffers shared across chunks during image decoding.
#[derive(Default)]
pub(crate) struct DecodeScratch {
    pub(crate) piz: PizScratch,
    pub(crate) tmp_buf: Vec<u8>,
    pub(crate) out_buf: Vec<u8>,
    #[cfg(feature = "dwa")]
    pub(crate) dwa: dwa::Context,
}

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

impl ChannelData {
    /// Number of scalar sample values held in this channel.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            ChannelData::Uint(values) => values.len(),
            ChannelData::Half(values) => values.len(),
            ChannelData::Float(values) => values.len(),
        }
    }

    /// Whether this channel holds zero values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The [`PixelType`] matching this channel's variant (`UINT`, `HALF`, or `FLOAT`).
    #[must_use]
    pub const fn pixel_type(&self) -> PixelType {
        match self {
            ChannelData::Uint(_) => PixelType::UINT,
            ChannelData::Half(_) => PixelType::HALF,
            ChannelData::Float(_) => PixelType::FLOAT,
        }
    }

    /// Borrows the underlying `u32` slice if this is [`ChannelData::Uint`].
    #[must_use]
    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        match self {
            ChannelData::Uint(values) => Some(values),
            ChannelData::Half(_) | ChannelData::Float(_) => None,
        }
    }

    /// Borrows the underlying `u16` half-float bit slice if this is [`ChannelData::Half`].
    #[must_use]
    pub fn as_u16_slice(&self) -> Option<&[u16]> {
        match self {
            ChannelData::Half(values) => Some(values),
            ChannelData::Uint(_) | ChannelData::Float(_) => None,
        }
    }

    /// Borrows the underlying `f32` slice if this is [`ChannelData::Float`].
    #[must_use]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            ChannelData::Float(values) => Some(values),
            ChannelData::Uint(_) | ChannelData::Half(_) => None,
        }
    }

    /// Returns a `Vec<f32>` of this channel's values (`Float` cloned as-is, `Half` converted
    /// via [`half_to_float`], `Uint` cast numerically with `as f32`).
    #[must_use]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            ChannelData::Float(values) => values.clone(),
            ChannelData::Half(values) => values.iter().copied().map(half_to_float).collect(),
            ChannelData::Uint(values) => values.iter().copied().map(|v| v as f32).collect(),
        }
    }

    pub(crate) fn as_view_mut(&mut self) -> ChannelViewMut<'_> {
        match self {
            ChannelData::Uint(values) => ChannelViewMut::Uint(values),
            ChannelData::Half(values) => ChannelViewMut::Half(values),
            ChannelData::Float(values) => ChannelViewMut::Float(values),
        }
    }
}

/// Mutable slice view into a [`ChannelData`] buffer, allowing safe non-overlapping row band
/// splits across scoped worker threads without `unsafe`.
pub(crate) enum ChannelViewMut<'a> {
    Uint(&'a mut [u32]),
    Half(&'a mut [u16]),
    Float(&'a mut [f32]),
}

#[cfg(feature = "std")]
impl<'a> ChannelViewMut<'a> {
    /// Splits off the first `count` elements from `self`, returning a mutable view over those
    /// `count` elements and advancing `self` to the remainder.
    pub(crate) fn take_front(&mut self, count: usize) -> ChannelViewMut<'a> {
        match core::mem::replace(self, ChannelViewMut::Float(&mut [])) {
            ChannelViewMut::Uint(slice) => {
                let (head, tail) = slice.split_at_mut(count);
                *self = ChannelViewMut::Uint(tail);
                ChannelViewMut::Uint(head)
            }
            ChannelViewMut::Half(slice) => {
                let (head, tail) = slice.split_at_mut(count);
                *self = ChannelViewMut::Half(tail);
                ChannelViewMut::Half(head)
            }
            ChannelViewMut::Float(slice) => {
                let (head, tail) = slice.split_at_mut(count);
                *self = ChannelViewMut::Float(tail);
                ChannelViewMut::Float(head)
            }
        }
    }
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
            pixel_data_size = pixel_data_size.checked_add(2)?;
            channel_offset = channel_offset.checked_add(2)?;
        } else if channel.pixel_type == PixelType::FLOAT || channel.pixel_type == PixelType::UINT {
            pixel_data_size = pixel_data_size.checked_add(4)?;
            channel_offset = channel_offset.checked_add(4)?;
        } else {
            return None;
        }
    }
    Some((channel_offset_list, pixel_data_size))
}

/// Decoded byte size per pixel across all channels (`2` for HALF->HALF, `4` otherwise).
pub(crate) fn decoded_bytes_per_pixel(
    channels: &[ChannelInfo],
    requested_pixel_types: &[PixelType],
) -> Option<usize> {
    if requested_pixel_types.len() != channels.len() {
        return None;
    }
    let mut total = 0usize;
    for (c, channel) in channels.iter().enumerate() {
        let bpp = if channel.pixel_type == PixelType::HALF {
            if requested_pixel_types[c] == PixelType::HALF {
                2usize
            } else if requested_pixel_types[c] == PixelType::FLOAT {
                4
            } else {
                return None;
            }
        } else if channel.pixel_type == PixelType::FLOAT || channel.pixel_type == PixelType::UINT {
            4
        } else {
            return None;
        };
        total = total.checked_add(bpp)?;
    }
    Some(total)
}

/// `AllocateImage`: room for `data_width` by `data_height` values of each channel, as the
/// type that was asked for.
pub(crate) fn allocate_image(
    channels: &[ChannelInfo],
    requested_pixel_types: &[PixelType],
    data_width: i32,
    data_height: i32,
) -> Option<Vec<ChannelData>> {
    if requested_pixel_types.len() != channels.len() {
        return None;
    }
    let mut images = Vec::with_capacity(channels.len());
    if data_width < 0 || data_height < 0 {
        return None;
    }
    let data_len = (data_width as usize).checked_mul(data_height as usize)?;

    for (c, channel) in channels.iter().enumerate() {
        if channel.pixel_type == PixelType::HALF {
            if requested_pixel_types[c] == PixelType::HALF {
                images.push(ChannelData::Half(zeros(data_len)?));
            } else if requested_pixel_types[c] == PixelType::FLOAT {
                images.push(ChannelData::Float(zeros(data_len)?));
            } else {
                return None;
            }
        } else if channel.pixel_type == PixelType::FLOAT {
            images.push(ChannelData::Float(zeros(data_len)?));
        } else if channel.pixel_type == PixelType::UINT {
            images.push(ChannelData::Uint(zeros(data_len)?));
        } else {
            return None;
        }
    }

    Some(images)
}

/// `len` zeros, or `None` where there is not the memory for them: a damaged file can ask for
/// more than there is, which the original answers with a message too.
pub(crate) fn zeros<T: Clone + Default>(len: usize) -> Option<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, T::default());
    Some(values)
}

/// Where the line `v` of a block and its value `u` go in a channel's values.
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

/// The halves of `src`, two bytes each, into consecutive places of a channel from `start`.
fn put_halves(image: &mut ChannelViewMut<'_>, start: usize, src: &[u8]) -> bool {
    let (src, _) = src.as_chunks::<2>();
    let fits = |len: usize| len.checked_sub(start).is_some_and(|room| room >= src.len());
    match image {
        ChannelViewMut::Half(values) => {
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
        ChannelViewMut::Float(values) => {
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
        ChannelViewMut::Uint(values) => {
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
fn put_fours(image: &mut ChannelViewMut<'_>, start: usize, src: &[u8]) -> bool {
    let (src, _) = src.as_chunks::<4>();
    let fits = |len: usize| len.checked_sub(start).is_some_and(|room| room >= src.len());
    match image {
        ChannelViewMut::Uint(values) => {
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
        ChannelViewMut::Float(values) => {
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
        ChannelViewMut::Half(_) => src.is_empty(),
    }
}

/// Unpacks scanline-interleaved decompressed channel data from `out_buf` into `out_images`.
#[allow(clippy::too_many_arguments)]
fn unpack_lines(
    out_images: &mut [ChannelViewMut<'_>],
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
                return false;
            }
        }
    }
    true
}

/// The size of a decompressed block of `width` by `num_lines` pixels of `pixel_data_size`
/// bytes, or `None` where that is not a size.
fn block_size(width: i32, num_lines: i32, pixel_data_size: usize) -> Option<usize> {
    if width < 0 || num_lines < 0 {
        return None;
    }
    (width as usize)
        .checked_mul(num_lines as usize)?
        .checked_mul(pixel_data_size)
        .filter(|&size| size <= isize::MAX as usize / 2)
}

/// `DecodePixelData`: decodes one scanline or tile chunk into `out_images` using `scratch`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn decode_pixel_data_with_scratch(
    out_images: &mut [ChannelViewMut<'_>],
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
    scratch: &mut DecodeScratch,
) -> bool {
    if requested_pixel_types.len() != channels.len() || out_images.len() != channels.len() {
        return false;
    }
    let block = &data[..data_len];
    if compression_type == Compression::PIZ {
        if width == 0 || num_lines == 0 || pixel_data_size == 0 {
            return false;
        }
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
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
        scratch.out_buf.clear();
        scratch.out_buf.resize(size, 0);
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
        if !decompress_piz_with_scratch(
            &mut scratch.out_buf[..size],
            data,
            size,
            data_len,
            &channel_sizes,
            width,
            num_lines,
            &mut scratch.piz,
        ) {
            return false;
        }
        unpack_lines(
            out_images,
            requested_pixel_types,
            &scratch.out_buf[..size],
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
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
        if size == 0 {
            return false;
        }
        scratch.out_buf.clear();
        scratch.out_buf.resize(size, 0);
        if decompress_zip_with_scratch(&mut scratch.out_buf[..size], block, &mut scratch.tmp_buf)
            .is_none()
        {
            return false;
        }
        unpack_lines(
            out_images,
            requested_pixel_types,
            &scratch.out_buf[..size],
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
        #[cfg(not(feature = "zfp"))]
        {
            let _ = zfp;
            false
        }
        #[cfg(feature = "zfp")]
        {
            let Some(zfp) = zfp else { return false };
            let Some(size) = block_size(width, num_lines, pixel_data_size) else {
                return false;
            };
            if size == 0 {
                return false;
            }
            scratch.out_buf.clear();
            scratch.out_buf.resize(size, 0);
            if decompress_zfp(
                &mut scratch.out_buf[..size],
                width,
                num_lines,
                channels.len(),
                block,
                zfp,
            )
            .is_none()
            {
                return false;
            }

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
                        &scratch.out_buf[line_ptr..line_ptr + 4 * width],
                    ) {
                        return false;
                    }
                }
            }
            true
        }
    } else if compression_type == Compression::RLE {
        let Some(size) = block_size(width, num_lines, pixel_data_size) else {
            return false;
        };
        if size == 0 {
            return false;
        }
        scratch.out_buf.clear();
        scratch.out_buf.resize(size, 0);
        if !decompress_rle_with_scratch(
            &mut scratch.out_buf[..size],
            size,
            block,
            &mut scratch.tmp_buf,
        ) {
            return false;
        }
        unpack_lines(
            out_images,
            requested_pixel_types,
            &scratch.out_buf[..size],
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
        let Some(size) = block_size(width, num_lines, pixel_data_size).filter(|&size| size != 0)
        else {
            return false;
        };
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
        scratch.out_buf.clear();
        scratch.out_buf.resize(size, 0);

        let pixel_types: Vec<PixelType> =
            channels.iter().map(|channel| channel.pixel_type).collect();
        if !decompress_pxr24_with_scratch(
            &mut scratch.out_buf[..size],
            block,
            width,
            num_lines,
            &pixel_types,
            as_tinyexr,
            &mut scratch.tmp_buf,
        ) {
            return false;
        }

        unpack_lines(
            out_images,
            requested_pixel_types,
            &scratch.out_buf[..size],
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
        let mut b44_out_size = 0usize;
        for channel in channels {
            let Some(cw) = sampled_dim(width, channel.x_sampling) else {
                return false;
            };
            let Some(ch) = sampled_dim(num_lines, channel.y_sampling) else {
                return false;
            };
            let bpp = if channel.pixel_type == PixelType::HALF {
                2usize
            } else {
                4
            };
            let Some(added) = cw
                .checked_mul(ch)
                .and_then(|n| n.checked_mul(bpp))
                .and_then(|n| b44_out_size.checked_add(n))
            else {
                return false;
            };
            b44_out_size = added;
        }

        if b44_out_size == 0 || b44_out_size > isize::MAX as usize / 2 {
            return false;
        }
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
        scratch.out_buf.clear();
        scratch.out_buf.resize(b44_out_size, 0);

        if !decompress_b44(
            &mut scratch.out_buf[..b44_out_size],
            block,
            width,
            num_lines,
            channels,
        ) {
            return false;
        }

        let mut ch_offset = 0usize;
        for (c, channel) in channels.iter().enumerate() {
            let Some(ch_width) = sampled_dim(width, channel.x_sampling) else {
                return false;
            };
            let Some(ch_height) = sampled_dim(num_lines, channel.y_sampling) else {
                return false;
            };
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
                        &scratch.out_buf[line_ptr..line_ptr + 2 * ch_width],
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
                        &scratch.out_buf[line_ptr..line_ptr + 4 * ch_width],
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
        #[cfg(not(feature = "dwa"))]
        {
            false
        }
        #[cfg(feature = "dwa")]
        {
            let Some(size) =
                block_size(width, num_lines, pixel_data_size).filter(|&size| size != 0)
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
            scratch.out_buf.clear();
            scratch.out_buf.resize(size, 0);
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
            if scratch
                .dwa
                .decompress_chunk(
                    block,
                    &dwa_channels,
                    width as usize,
                    num_lines as usize,
                    &mut scratch.out_buf[..size],
                )
                .is_err()
            {
                return false;
            }
            unpack_lines(
                out_images,
                requested_pixel_types,
                &scratch.out_buf[..size],
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
        }
    } else if compression_type == Compression::NONE {
        let width_size = width.max(0) as usize;
        for (c, channel) in channels.iter().enumerate() {
            for v in 0..num_lines.max(0) as usize {
                let line_ptr = v
                    .wrapping_mul(pixel_data_size)
                    .wrapping_mul(width as isize as usize)
                    .wrapping_add(channel_offset_list[c].wrapping_mul(width as isize as usize));
                let out_line = image_index(line_order, height, y, x_stride, v, 0);

                if channel.pixel_type == PixelType::HALF {
                    if requested_pixel_types[c] == PixelType::HALF {
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
        true
    }
}

/// `DecodeTiledPixelData`: a tile's block into the tile's channels using `scratch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_tiled_pixel_data_with_scratch(
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
    scratch: &mut DecodeScratch,
) -> Option<(i32, i32)> {
    if tile_size_x.wrapping_mul(tile_offset_x) > data_width
        || tile_size_y.wrapping_mul(tile_offset_y) > data_height
    {
        return None;
    }

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

    let mut views: Vec<ChannelViewMut<'_>> = out_images
        .iter_mut()
        .map(ChannelData::as_view_mut)
        .collect();
    decode_pixel_data_with_scratch(
        &mut views,
        requested_pixel_types,
        data,
        data_len,
        compression_type,
        LineOrder::INCREASING_Y,
        width,
        tile_size_y,
        tile_size_x,
        0,
        0,
        height,
        pixel_data_size,
        channels,
        channel_offset_list,
        zfp,
        as_tinyexr,
        scratch,
    )
    .then_some((width, height))
}
