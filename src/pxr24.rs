//! The PXR24 compression of OpenEXR: a port of `DecompressPxr24` and `CompressPxr24` of
//! tinyexr v3.4.0 (`tinyexr.h`).
//!
//! Floats are kept as 24 bits, which loses a little; halves and integers are kept whole. The
//! bytes of a line's values are stored most significant first, as differences from the value
//! to the left, and the whole block is a zlib stream.

use alloc::vec::Vec;
use miniz_nrp as miniz;

use super::PixelType;

/// `DecompressPxr24` reusing `pxr24_buf` across chunks.
pub(crate) fn decompress_pxr24_with_scratch(
    out: &mut [u8],
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    pixel_types: &[PixelType],
    as_tinyexr: bool,
    pxr24_buf: &mut Vec<u8>,
) -> bool {
    if data_width < 0 || num_lines < 0 {
        return false;
    }
    let Some(ch_pixels) = (data_width as usize).checked_mul(num_lines as usize) else {
        return false;
    };

    // Calculate the expected uncompressed size in PXR24 format without `i32` wrap.
    let mut pxr24_size = 0usize;
    for &pixel_type in pixel_types {
        let bpp = if pixel_type == PixelType::UINT {
            4usize
        } else if pixel_type == PixelType::HALF {
            2
        } else {
            // FLOAT: 24 bits
            3
        };
        let Some(added) = ch_pixels
            .checked_mul(bpp)
            .and_then(|n| pxr24_size.checked_add(n))
        else {
            return false;
        };
        pxr24_size = added;
    }

    // First decompress with zlib.
    if as_tinyexr && pxr24_size == input.len() {
        pxr24_buf.clear();
        pxr24_buf.extend_from_slice(input);
    } else if miniz::decompress_into(input, miniz::Format::Zlib, pxr24_size, pxr24_buf).is_err() {
        return false;
    }

    if pxr24_buf.len() != pxr24_size {
        return false;
    }

    // Convert from PXR24 format to standard EXR format.
    let mut in_p = 0usize;
    let mut out_p = 0usize;
    let w = data_width as usize;

    for _line in 0..num_lines as usize {
        for &pixel_type in pixel_types {
            if pixel_type == PixelType::UINT {
                // UINT: 4 byte planes with delta encoding.
                let (ptr0, ptr1, ptr2, ptr3) = (in_p, in_p + w, in_p + w * 2, in_p + w * 3);
                in_p += w * 4;

                let mut pixel = 0u32;
                for x in 0..w {
                    let diff = (u32::from(pxr24_buf[ptr0 + x]) << 24)
                        | (u32::from(pxr24_buf[ptr1 + x]) << 16)
                        | (u32::from(pxr24_buf[ptr2 + x]) << 8)
                        | u32::from(pxr24_buf[ptr3 + x]);
                    pixel = pixel.wrapping_add(diff);
                    out[out_p..out_p + 4].copy_from_slice(&pixel.to_le_bytes());
                    out_p += 4;
                }
            } else if pixel_type == PixelType::HALF {
                // HALF: 2 byte planes with delta encoding.
                let (ptr0, ptr1) = (in_p, in_p + w);
                in_p += w * 2;

                let mut pixel = 0u32;
                for x in 0..w {
                    let diff =
                        (u32::from(pxr24_buf[ptr0 + x]) << 8) | u32::from(pxr24_buf[ptr1 + x]);
                    pixel = pixel.wrapping_add(diff);
                    out[out_p..out_p + 2].copy_from_slice(&(pixel as u16).to_le_bytes());
                    out_p += 2;
                }
            } else {
                // FLOAT: 3 byte planes (24-bit) with delta encoding.
                let (ptr0, ptr1, ptr2) = (in_p, in_p + w, in_p + w * 2);
                in_p += w * 3;

                let mut pixel = 0u32;
                for x in 0..w {
                    let diff = (u32::from(pxr24_buf[ptr0 + x]) << 24)
                        | (u32::from(pxr24_buf[ptr1 + x]) << 16)
                        | (u32::from(pxr24_buf[ptr2 + x]) << 8);
                    pixel = pixel.wrapping_add(diff);
                    // Float24 to float32: the 24-bit value is in the upper 24 bits.
                    out[out_p..out_p + 4].copy_from_slice(&pixel.to_le_bytes());
                    out_p += 4;
                }
            }
        }
    }

    true
}

/// `DecompressPxr24`: a block of `num_lines` lines of `data_width` pixels into `out`, four
/// bytes a value for integers and floats and two for halves. `pixel_types` are the channels'.
#[allow(dead_code)]
pub(crate) fn decompress_pxr24(
    out: &mut [u8],
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    pixel_types: &[PixelType],
    as_tinyexr: bool,
) -> bool {
    let mut pxr24_buf = Vec::new();
    decompress_pxr24_with_scratch(
        out,
        input,
        data_width,
        num_lines,
        pixel_types,
        as_tinyexr,
        &mut pxr24_buf,
    )
}

/// `float_to_float24`: a float as the 24 bits PXR24 keeps of it, rounded to nearest; an
/// infinity stays one and a NaN stays one.
// (The names are tinyexr's, so that the code reads against the original.)
#[allow(clippy::many_single_char_names)]
#[inline]
fn float_to_float24(f: f32) -> u32 {
    let i = f.to_bits();
    let s = i & 0x8000_0000;
    let e = i & 0x7f80_0000;
    let mut m = i & 0x007f_ffff;

    if e == 0x7f80_0000 {
        if m != 0 {
            m >>= 8;
            return (s >> 8) | (e >> 8) | m | u32::from(m == 0);
        }
        return (s >> 8) | (e >> 8);
    }

    let mut i = ((e | m) + (m & 0x0000_0080)) >> 8;

    if i >= 0x007f_8000 {
        i = (e | m) >> 8;
    }

    (s >> 8) | i
}

/// `CompressPxr24` appending into `out_data`, reusing `pxr24_buf` and `compressor` across
/// chunks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compress_pxr24_into(
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    file_types: &[PixelType],
    as_tinyexr: bool,
    pxr24_buf: &mut Vec<u8>,
    compressor: &mut miniz::Compressor,
    out_data: &mut Vec<u8>,
) {
    let w = data_width.max(0) as usize;
    let h = num_lines.max(0) as usize;
    let ch_pixels = w * h;
    let mut pxr24_size = 0usize;
    for &file_type in file_types {
        if file_type == PixelType::UINT {
            pxr24_size += ch_pixels * 4;
        } else if file_type == PixelType::HALF {
            pxr24_size += ch_pixels * 2;
        } else {
            pxr24_size += ch_pixels * 3;
        }
    }

    pxr24_buf.clear();
    pxr24_buf.resize(pxr24_size, 0);
    let mut in_p = 0usize;
    let mut out_p = 0usize;

    for _line in 0..h {
        for &file_type in file_types {
            if file_type == PixelType::UINT {
                let ptr = out_p;
                out_p += w * 4;

                let mut prev_pixel = 0u32;
                for x in 0..w {
                    let pixel = u32::from_le_bytes([
                        input[in_p],
                        input[in_p + 1],
                        input[in_p + 2],
                        input[in_p + 3],
                    ]);
                    in_p += 4;

                    let diff = pixel.wrapping_sub(prev_pixel);
                    prev_pixel = pixel;

                    pxr24_buf[ptr + x] = (diff >> 24) as u8;
                    pxr24_buf[ptr + w + x] = (diff >> 16) as u8;
                    pxr24_buf[ptr + w * 2 + x] = (diff >> 8) as u8;
                    pxr24_buf[ptr + w * 3 + x] = diff as u8;
                }
            } else if file_type == PixelType::HALF {
                let ptr = out_p;
                out_p += w * 2;

                let mut prev_pixel = 0u32;
                for x in 0..w {
                    let pixel = u32::from(u16::from_le_bytes([input[in_p], input[in_p + 1]]));
                    in_p += 2;

                    let diff = pixel.wrapping_sub(prev_pixel);
                    prev_pixel = pixel;

                    pxr24_buf[ptr + x] = (diff >> 8) as u8;
                    pxr24_buf[ptr + w + x] = diff as u8;
                }
            } else {
                let ptr = out_p;
                out_p += w * 3;

                let mut prev_pixel = 0u32;
                for x in 0..w {
                    let f = f32::from_bits(u32::from_le_bytes([
                        input[in_p],
                        input[in_p + 1],
                        input[in_p + 2],
                        input[in_p + 3],
                    ]));
                    in_p += 4;

                    let pixel24 = float_to_float24(f);
                    let diff = pixel24.wrapping_sub(prev_pixel);
                    prev_pixel = pixel24;

                    pxr24_buf[ptr + x] = (diff >> 16) as u8;
                    pxr24_buf[ptr + w + x] = (diff >> 8) as u8;
                    pxr24_buf[ptr + w * 2 + x] = diff as u8;
                }
            }
        }
    }

    compressor.reset_with(&miniz::Options::default());
    compressor
        .compress(pxr24_buf, miniz::Flush::Finish)
        .expect("freshly reset compressor");
    let block = compressor.output();
    out_data.extend_from_slice(if !as_tinyexr && block.len() >= input.len() {
        input
    } else {
        block
    });
}

/// `CompressPxr24`: standalone helper.
#[allow(dead_code)]
pub(crate) fn compress_pxr24(
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    file_types: &[PixelType],
) -> Vec<u8> {
    let mut pxr24_buf = Vec::new();
    let mut compressor = miniz::Compressor::default();
    let mut out = Vec::new();
    compress_pxr24_into(
        input,
        data_width,
        num_lines,
        file_types,
        true,
        &mut pxr24_buf,
        &mut compressor,
        &mut out,
    );
    out
}
