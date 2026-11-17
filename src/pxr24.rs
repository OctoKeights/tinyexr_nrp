//! The PXR24 compression of OpenEXR: a port of `DecompressPxr24` and `CompressPxr24` of
//! tinyexr v3.4.0 (`tinyexr.h`).
//!
//! Floats are kept as 24 bits, which loses a little; halves and integers are kept whole. The
//! bytes of a line's values are stored most significant first, as differences from the value
//! to the left, and the whole block is a zlib stream.

use miniz_nrp as miniz;

use super::PixelType;

/// `DecompressPxr24`: a block of `num_lines` lines of `data_width` pixels into `out`, four
/// bytes a value for integers and floats and two for halves. `pixel_types` are the channels'.
///
/// The original takes a block that is as long as its values are in PXR24's own layout for
/// one that was not compressed, which no writer makes, its own included; a compressed block
/// that comes out that long is then read as other values. That is done only `as_tinyexr`.
pub(crate) fn decompress_pxr24(
    out: &mut [u8],
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    pixel_types: &[PixelType],
    as_tinyexr: bool,
) -> bool {
    // PXR24 stores data per-scanline, per-channel with byte planes: UINT in 4 bytes, HALF in
    // 2 bytes, FLOAT in 3 bytes (24-bit).

    // Calculate the expected uncompressed size in PXR24 format.
    let mut pxr24_size = 0usize;
    for &pixel_type in pixel_types {
        let ch_pixels = data_width.wrapping_mul(num_lines);
        if pixel_type == PixelType::UINT {
            pxr24_size += (ch_pixels as usize).wrapping_mul(4);
        } else if pixel_type == PixelType::HALF {
            pxr24_size += (ch_pixels as usize).wrapping_mul(2);
        } else {
            // FLOAT: 24 bits
            pxr24_size += (ch_pixels as usize).wrapping_mul(3);
        }
    }

    // First decompress with zlib.
    let pxr24_buf;
    if as_tinyexr && pxr24_size == input.len() {
        // Data is not compressed.
        pxr24_buf = input.to_vec();
    } else {
        match miniz::decompress(input, pxr24_size) {
            Ok(buf) => pxr24_buf = buf,
            Err(_) => return false,
        }
    }

    if pxr24_buf.len() != pxr24_size {
        return false;
    }

    // Convert from PXR24 format to standard EXR format.
    let mut in_p = 0usize;
    let mut out_p = 0usize;
    let w = data_width as usize;

    for _line in 0..num_lines {
        for &pixel_type in pixel_types {
            if pixel_type == PixelType::UINT {
                // UINT: 4 byte planes with delta encoding.
                let (ptr0, ptr1, ptr2, ptr3) = (in_p, in_p + w, in_p + w * 2, in_p + w * 3);
                in_p += w * 4;

                let mut pixel = 0u32;
                for x in 0..w {
                    let diff = ((pxr24_buf[ptr0 + x] as u32) << 24)
                        | ((pxr24_buf[ptr1 + x] as u32) << 16)
                        | ((pxr24_buf[ptr2 + x] as u32) << 8)
                        | pxr24_buf[ptr3 + x] as u32;
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
                    let diff = ((pxr24_buf[ptr0 + x] as u32) << 8) | pxr24_buf[ptr1 + x] as u32;
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
                    let diff = ((pxr24_buf[ptr0 + x] as u32) << 24)
                        | ((pxr24_buf[ptr1 + x] as u32) << 16)
                        | ((pxr24_buf[ptr2 + x] as u32) << 8);
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

/// `float_to_float24`: a float as the 24 bits PXR24 keeps of it, rounded to nearest; an
/// infinity stays one and a NaN stays one.
#[inline]
fn float_to_float24(f: f32) -> u32 {
    let i = f.to_bits();
    // Disassemble the 32-bit floating point number, f, into sign, s, exponent, e, and
    // significand, m.
    let s = i & 0x80000000;
    let e = i & 0x7f800000;
    let mut m = i & 0x007fffff;

    if e == 0x7f800000 {
        if m != 0 {
            // F is a NAN; we preserve the sign bit and the 15 leftmost bits of the
            // significand, with one exception: if the 15 leftmost bits are all zero, the NAN
            // would turn into an infinity, so we have to set at least one bit in the
            // significand.
            m >>= 8;
            return (s >> 8) | (e >> 8) | m | u32::from(m == 0);
        } else {
            // F is an infinity.
            return (s >> 8) | (e >> 8);
        }
    }

    // F is finite, round the significand to 15 bits.
    let mut i = ((e | m) + (m & 0x00000080)) >> 8;

    if i >= 0x7f8000 {
        // F was close to FLT_MAX, and the significand was rounded up, resulting in an
        // exponent overflow. Avoid the overflow by truncating the significand instead of
        // rounding it.
        i = (e | m) >> 8;
    }

    (s >> 8) | i
}

/// `CompressPxr24`: a block of `num_lines` lines of `data_width` pixels, whose channels are
/// stored as `file_types`, as a PXR24 block: floats cut to 24 bits, every value as its
/// difference from the one before it in its line, the bytes of the values of a line apart
/// (all first bytes, then all second ones), and the whole as a zlib stream.
pub(crate) fn compress_pxr24(
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    file_types: &[PixelType],
) -> Vec<u8> {
    // Calculate PXR24 buffer size.
    // PXR24 converts FLOAT to 24-bit, HALF stays 16-bit, UINT stays 32-bit.
    let w = data_width as usize;
    let mut pxr24_size = 0usize;
    for &file_type in file_types {
        let ch_pixels = (data_width * num_lines) as usize;
        if file_type == PixelType::UINT {
            pxr24_size += ch_pixels * 4;
        } else if file_type == PixelType::HALF {
            pxr24_size += ch_pixels * 2;
        } else {
            pxr24_size += ch_pixels * 3;
        }
    }

    let mut pxr24_buf = vec![0u8; pxr24_size];
    let mut in_p = 0usize;
    let mut out_p = 0usize;

    // Process each scanline.
    for _line in 0..num_lines {
        // Process each channel.
        for &file_type in file_types {
            if file_type == PixelType::UINT {
                // UINT: 4 bytes, store as 4 separate byte planes with delta encoding.
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
                // HALF: 2 bytes, store as 2 separate byte planes with delta encoding.
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
                // FLOAT: convert to 24-bit, store as 3 separate byte planes with delta
                // encoding.
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

    // Compress with zlib.
    miniz::compress(&pxr24_buf, miniz::Level::DEFAULT)
}
