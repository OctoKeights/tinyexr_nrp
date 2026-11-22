//! The B44 and B44A compression of OpenEXR: a port of `DecompressB44`, `CompressB44` and
//! what is under them in tinyexr v3.4.0 (`tinyexr.h`).
//!
//! Halves are packed four by four: a block of sixteen takes fourteen bytes, or three where
//! all sixteen are the same (which B44A writes and both read). Floats and integers are
//! stored as they are.

use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::{ChannelInfo, PixelType};

/// `B44HalfToFloat`.
// (The names are tinyexr's, so that the code reads against the original.)
#[allow(clippy::many_single_char_names)]
fn b44_half_to_float(h: u16) -> f32 {
    let s = u32::from((h >> 15) & 0x1);
    let e = u32::from((h >> 10) & 0x1f);
    let m = u32::from(h & 0x3ff);

    if e == 0 {
        if m == 0 {
            return f32::from_bits(s << 31);
        }
        // Denormalized
        let mut f = m as f32 / 1024.0;
        f *= 1.0 / 16384.0;
        return if s != 0 { -f } else { f };
    } else if e == 31 {
        return f32::from_bits((s << 31) | 0x7f80_0000 | (m << 13));
    }

    f32::from_bits((s << 31) | ((e + 112) << 23) | (m << 13))
}

/// `B44FloatToHalf`.
// (The names are tinyexr's, so that the code reads against the original.)
#[allow(clippy::many_single_char_names)]
fn b44_float_to_half(f: f32) -> u16 {
    let i = f.to_bits();
    let s = ((i >> 31) & 0x1) as i32;
    let e = ((i >> 23) & 0xff) as i32;
    let mut m = (i & 0x007f_ffff) as i32;

    if e == 0 {
        return (s << 15) as u16;
    } else if e == 255 {
        return ((s << 15) | 0x7c00 | (m >> 13)) as u16;
    } else if e < 113 {
        if e < 103 {
            return (s << 15) as u16;
        }
        m = (m | 0x0080_0000) >> (114 - e);
        return ((s << 15) | (m >> 13)) as u16;
    } else if e > 142 {
        return ((s << 15) | 0x7c00) as u16;
    }

    ((s << 15) | ((e - 112) << 10) | (m >> 13)) as u16
}

/// Pure-Rust IEEE-754 natural logarithm for positive finite `f64` values (`no_std` fallback).
#[cfg(any(not(feature = "std"), test))]
fn ln_f64_pure(v: f64) -> f64 {
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let bits = v.to_bits();
    let mut exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let poly = 1.0
        + t2 * (1.0 / 3.0
            + t2 * (1.0 / 5.0
                + t2 * (1.0 / 7.0
                    + t2 * (1.0 / 9.0
                        + t2 * (1.0 / 11.0
                            + t2 * (1.0 / 13.0 + t2 * (1.0 / 15.0 + t2 * (1.0 / 17.0))))))));
    let ln_m = 2.0 * t * poly;
    f64::from(exp) * LN2_HI + (ln_m + f64::from(exp) * LN2_LO)
}

#[cfg(any(not(feature = "std"), test))]
fn convert_to_linear_pure(x: u16) -> u16 {
    if (x & 0x7c00) == 0x7c00 || x > 0x8000 {
        0
    } else {
        let f = b44_half_to_float(x);
        if f <= 0.0 {
            0
        } else {
            b44_float_to_half((8.0 * ln_f64_pure(f64::from(f))) as f32)
        }
    }
}

/// The original's `g_b44_log_table`, which `InitB44Tables` fills for the channels a file
/// marks as linear (`pLinear`), whose halves B44 converts via `8 * ln(f)` on decode.
/// (`g_b44_exp_table` is never used by tinyexr and is omitted.)
#[cfg(feature = "std")]
fn b44_log_table() -> &'static [u16] {
    static LOG_TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    LOG_TABLE.get_or_init(|| {
        let mut log_table = vec![0u16; 65536];
        for (i, slot) in log_table.iter_mut().enumerate() {
            let x = i as u16;
            if (x & 0x7c00) == 0x7c00 || x > 0x8000 {
                *slot = 0;
            } else {
                let f = b44_half_to_float(x);
                if f <= 0.0 {
                    *slot = 0;
                } else {
                    *slot = b44_float_to_half((8.0 * f64::from(f).ln()) as f32);
                }
            }
        }
        log_table
    })
}

#[inline]
fn apply_p_linear(block: &mut [u16; 16]) {
    #[cfg(feature = "std")]
    {
        let log_table = b44_log_table();
        for value in block {
            *value = log_table[*value as usize];
        }
    }
    #[cfg(not(feature = "std"))]
    {
        for value in block {
            *value = convert_to_linear_pure(*value);
        }
    }
}

/// `UnpackB44Block`: sixteen halves out of fourteen bytes.
fn unpack_b44_block(dst: &mut [u16; 16], src: &[u8]) {
    let s = |i: usize| u32::from(src[i]);
    let s0 = ((s(0) << 8) | s(1)) as u16;

    let shift = s(2) >> 2;
    let bias = u32::from((0x20u32 << shift) as u16);
    // A value from the one it leans on and a difference of six bits.
    let step = |from: u16, six_bits: u32| {
        u32::from(from)
            .wrapping_add((six_bits & 0x3f).wrapping_mul(1u32 << shift))
            .wrapping_sub(bias) as u16
    };

    let s4 = step(s0, (s(2) << 4) | (s(3) >> 4));
    let s8 = step(s4, (s(3) << 2) | (s(4) >> 6));
    let s12 = step(s8, s(4));

    let s1 = step(s0, s(5) >> 2);
    let s5 = step(s4, (s(5) << 4) | (s(6) >> 4));
    let s9 = step(s8, (s(6) << 2) | (s(7) >> 6));
    let s13 = step(s12, s(7));

    let s2 = step(s1, s(8) >> 2);
    let s6 = step(s5, (s(8) << 4) | (s(9) >> 4));
    let s10 = step(s9, (s(9) << 2) | (s(10) >> 6));
    let s14 = step(s13, s(10));

    let s3 = step(s2, s(11) >> 2);
    let s7 = step(s6, (s(11) << 4) | (s(12) >> 4));
    let s11 = step(s10, (s(12) << 2) | (s(13) >> 6));
    let s15 = step(s14, s(13));

    *dst = [
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
    ];

    // Convert from the ordered-magnitude representation back to halves.
    for value in dst.iter_mut() {
        if *value & 0x8000 != 0 {
            *value &= 0x7fff;
        } else {
            *value = !*value;
        }
    }
}

/// `UnpackB44FlatBlock`: sixteen halves, all the same, out of three bytes.
fn unpack_b44_flat_block(dst: &mut [u16; 16], src: &[u8]) {
    let t = (u16::from(src[0]) << 8) | u16::from(src[1]);
    let h = if t & 0x8000 != 0 { t & 0x7fff } else { !t };
    *dst = [h; 16];
}

/// Subsampled dimension `ceil(dim / max(step, 1))` computed in `i64` so large `step` values
/// near `i32::MAX` never wrap negative.
#[inline]
pub(crate) fn sampled_dim(dim: i32, step: i32) -> Option<usize> {
    if dim < 0 {
        return None;
    }
    let s = if step > 0 { i64::from(step) } else { 1 };
    let d = (i64::from(dim) + s - 1) / s;
    usize::try_from(d).ok()
}

/// `DecompressB44`: a block of `num_lines` lines of `data_width` pixels into `out`, channel
/// after channel.
// (One pass, step for step as tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_lines)]
pub(crate) fn decompress_b44(
    out: &mut [u8],
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    channels: &[ChannelInfo],
) -> bool {
    if data_width < 0 || num_lines < 0 {
        return false;
    }

    // Guard: reject if the declared output would overflow the buffer.
    {
        let mut expected_out = 0usize;
        for channel in channels {
            let Some(cw) = sampled_dim(data_width, channel.x_sampling) else {
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
            let Some(pixels) = cw.checked_mul(ch) else {
                return false;
            };
            let Some(ch_total) = pixels.checked_mul(bpp) else {
                return false;
            };
            let Some(sum) = expected_out.checked_add(ch_total) else {
                return false;
            };
            expected_out = sum;
        }
        if expected_out > out.len() {
            return false;
        }
    }

    let mut in_p = 0usize;
    let in_end = input.len();

    // B44 processes each channel separately in 4x4 blocks.
    let mut scratch_buffers: Vec<Vec<u16>> = vec![Vec::new(); channels.len()];
    // Where a channel that is not of halves lies in the input, and its size.
    let mut nonhalf = vec![(0usize, 0usize); channels.len()];

    for (c, channel) in channels.iter().enumerate() {
        let Some(ch_width) = sampled_dim(data_width, channel.x_sampling) else {
            return false;
        };
        let Some(ch_height) = sampled_dim(num_lines, channel.y_sampling) else {
            return false;
        };

        if channel.pixel_type != PixelType::HALF {
            // B44 only compresses HALF; other types are stored uncompressed.
            let bpp = if channel.pixel_type == PixelType::UINT
                || channel.pixel_type == PixelType::FLOAT
            {
                4usize
            } else {
                2usize
            };
            let Some(ch_bytes) = ch_width
                .checked_mul(ch_height)
                .and_then(|n| n.checked_mul(bpp))
            else {
                return false;
            };
            match in_p.checked_add(ch_bytes) {
                Some(end) if end <= in_end => {}
                _ => return false,
            }
            nonhalf[c] = (in_p, ch_bytes);
            in_p += ch_bytes;
            continue;
        }

        // Pad to 4x4 block boundaries.
        let num_blocks_x = ch_width.div_ceil(4);
        let num_blocks_y = ch_height.div_ceil(4);
        let Some(padded_width) = num_blocks_x.checked_mul(4) else {
            return false;
        };
        let Some(padded_height) = num_blocks_y.checked_mul(4) else {
            return false;
        };
        let Some(scratch_len) = padded_width.checked_mul(padded_height) else {
            return false;
        };

        scratch_buffers[c].resize(scratch_len, 0);

        for by in 0..num_blocks_y {
            for bx in 0..num_blocks_x {
                let mut block = [0u16; 16];

                if in_p.checked_add(3).is_none_or(|end| end > in_end) {
                    return false;
                }

                // Check if it is a flat block (B44A feature, also in B44).
                if input[in_p + 2] >= (13 << 2) {
                    unpack_b44_flat_block(&mut block, &input[in_p..in_p + 3]);
                    in_p += 3;
                } else {
                    if in_p.checked_add(14).is_none_or(|end| end > in_end) {
                        return false;
                    }
                    unpack_b44_block(&mut block, &input[in_p..in_p + 14]);
                    in_p += 14;
                }

                // Apply pLinear conversion if needed.
                if channel.p_linear != 0 {
                    apply_p_linear(&mut block);
                }

                // Copy block to scratch buffer.
                for dy in 0..4 {
                    let y = by * 4 + dy;
                    for dx in 0..4 {
                        let x = bx * 4 + dx;
                        scratch_buffers[c][y * padded_width + x] = block[dy * 4 + dx];
                    }
                }
            }
        }
    }

    // Copy from scratch buffers to output (channel by channel, row by row).
    let mut out_p = 0usize;

    for (c, channel) in channels.iter().enumerate() {
        let Some(ch_width) = sampled_dim(data_width, channel.x_sampling) else {
            return false;
        };
        let Some(ch_height) = sampled_dim(num_lines, channel.y_sampling) else {
            return false;
        };
        let padded_width = ch_width.div_ceil(4) * 4;

        if channel.pixel_type == PixelType::HALF {
            for y in 0..ch_height {
                for x in 0..ch_width {
                    let val = scratch_buffers[c][y * padded_width + x];
                    out[out_p..out_p + 2].copy_from_slice(&val.to_le_bytes());
                    out_p += 2;
                }
            }
        } else if channel.pixel_type == PixelType::UINT || channel.pixel_type == PixelType::FLOAT {
            // The raw bytes were already range-checked above.
            let (start, size) = nonhalf[c];
            if size > 0 {
                out[out_p..out_p + size].copy_from_slice(&input[start..start + size]);
                out_p += size;
            }
        }
    }

    true
}

/// `B44ShiftAndRound`.
#[inline]
fn b44_shift_and_round(mut x: i32, mut shift: i32) -> i32 {
    // Compute y = x >> shift, rounding to nearest (ties to even).
    x <<= 1;
    let a = (1 << shift) - 1;
    shift += 1;
    let b = (x >> shift) & 1;
    (x + a + b) >> shift
}

/// `PackB44Block`: sixteen halves, a block of four by four, into 14 bytes of `out`, or into
/// 3 when they are all the same and `flatfields` allows it. The answer is how many.
fn pack_b44_block(out: &mut [u8], block: &[u16; 16], flatfields: bool, exactmax: bool) -> usize {
    const BIAS: i32 = 0x20;
    let mut d = [0i32; 16];
    let mut r = [0i32; 15];
    let mut r_min;
    let mut r_max;
    let mut t = [0u16; 16];
    let mut shift = -1i32;

    // Convert to ordered representation: handle NaN/Inf, then convert signed magnitude to
    // an ordered unsigned.
    for i in 0..16 {
        if (block[i] & 0x7c00) == 0x7c00 {
            t[i] = 0x8000; // NaN/Inf -> zero
        } else if block[i] & 0x8000 != 0 {
            t[i] = !block[i]; // Negative
        } else {
            t[i] = block[i] | 0x8000; // Positive
        }
    }

    // Find max.
    let mut t_max = 0u16;
    for &value in &t {
        if t_max < value {
            t_max = value;
        }
    }

    // Find the shift value that makes all differences fit in 6 bits.
    loop {
        shift += 1;

        for i in 0..16 {
            d[i] = b44_shift_and_round(i32::from(t_max) - i32::from(t[i]), shift);
        }

        r[0] = d[0] - d[4] + BIAS;
        r[1] = d[4] - d[8] + BIAS;
        r[2] = d[8] - d[12] + BIAS;

        r[3] = d[0] - d[1] + BIAS;
        r[4] = d[4] - d[5] + BIAS;
        r[5] = d[8] - d[9] + BIAS;
        r[6] = d[12] - d[13] + BIAS;

        r[7] = d[1] - d[2] + BIAS;
        r[8] = d[5] - d[6] + BIAS;
        r[9] = d[9] - d[10] + BIAS;
        r[10] = d[13] - d[14] + BIAS;

        r[11] = d[2] - d[3] + BIAS;
        r[12] = d[6] - d[7] + BIAS;
        r[13] = d[10] - d[11] + BIAS;
        r[14] = d[14] - d[15] + BIAS;

        r_min = r[0];
        r_max = r[0];
        for &value in &r[1..] {
            if r_min > value {
                r_min = value;
            }
            if r_max < value {
                r_max = value;
            }
        }
        if !(r_min < 0 || r_max > 0x3f) {
            break;
        }
    }

    // Check for flat field (all pixels same value).
    if r_min == BIAS && r_max == BIAS && flatfields {
        out[0] = (t[0] >> 8) as u8;
        out[1] = t[0] as u8;
        out[2] = 0xfc;
        return 3;
    }

    if exactmax {
        // Adjust t[0] so the pixel with max value is represented accurately.
        t[0] = t_max.wrapping_sub((d[0] << shift) as u16);
    }

    // Pack into 14 bytes.
    out[0] = (t[0] >> 8) as u8;
    out[1] = t[0] as u8;

    out[2] = ((shift << 2) | (r[0] >> 4)) as u8;
    out[3] = ((r[0] << 4) | (r[1] >> 2)) as u8;
    out[4] = ((r[1] << 6) | r[2]) as u8;

    out[5] = ((r[3] << 2) | (r[4] >> 4)) as u8;
    out[6] = ((r[4] << 4) | (r[5] >> 2)) as u8;
    out[7] = ((r[5] << 6) | r[6]) as u8;

    out[8] = ((r[7] << 2) | (r[8] >> 4)) as u8;
    out[9] = ((r[8] << 4) | (r[9] >> 2)) as u8;
    out[10] = ((r[9] << 6) | r[10]) as u8;

    out[11] = ((r[11] << 2) | (r[12] >> 4)) as u8;
    out[12] = ((r[12] << 4) | (r[13] >> 2)) as u8;
    out[13] = ((r[13] << 6) | r[14]) as u8;

    14
}

/// `CompressB44`: a block of `num_lines` lines of `data_width` pixels, one channel after the
/// other, whose channels are stored as `file_types`: the channels of halves in blocks of four
/// by four of 14 bytes each (of 3 where a block is all one value, for B44A), the others as
/// they are.
pub(crate) fn compress_b44(
    input: &[u8],
    data_width: i32,
    num_lines: i32,
    file_types: &[PixelType],
    is_b44a: bool,
) -> Vec<u8> {
    let w = data_width.max(0) as usize;
    let h = num_lines.max(0) as usize;
    let num_blocks_x = w.div_ceil(4);
    let num_blocks_y = h.div_ceil(4);
    let pixels = w * h;

    // Calculate max output size.
    let mut max_size = 0usize;
    for &file_type in file_types {
        if file_type == PixelType::HALF {
            max_size += num_blocks_x * num_blocks_y * 14;
        } else if file_type == PixelType::UINT || file_type == PixelType::FLOAT {
            max_size += pixels * 4;
        } else {
            max_size += pixels * 2;
        }
    }

    let mut out_buf = vec![0u8; max_size];
    let mut out_p = 0usize;
    let mut in_offset = 0usize;

    // Process each channel.
    for &file_type in file_types {
        if file_type != PixelType::HALF {
            // Non-HALF channels: copy raw data.
            let mut ch_bytes = pixels;
            if file_type == PixelType::UINT || file_type == PixelType::FLOAT {
                ch_bytes *= 4;
            } else {
                ch_bytes *= 2;
            }
            out_buf[out_p..out_p + ch_bytes]
                .copy_from_slice(&input[in_offset..in_offset + ch_bytes]);
            out_p += ch_bytes;
            in_offset += ch_bytes;
            continue;
        }

        // HALF channel: compress 4x4 blocks.
        let half_at = |index: usize| {
            u16::from_le_bytes([
                input[in_offset + 2 * index],
                input[in_offset + 2 * index + 1],
            ])
        };

        for by in 0..num_blocks_y {
            for bx in 0..num_blocks_x {
                let mut block = [0u16; 16];

                // Extract the 4x4 block (with edge replication).
                for dy in 0..4 {
                    let y = by * 4 + dy;
                    let src_y = if y >= h { h - 1 } else { y };

                    for dx in 0..4 {
                        let x = bx * 4 + dx;
                        let src_x = if x >= w { w - 1 } else { x };

                        block[dy * 4 + dx] = half_at(src_y * w + src_x);
                    }
                }

                // Pack block.
                let bytes_written = pack_b44_block(&mut out_buf[out_p..], &block, is_b44a, true);
                out_p += bytes_written;
            }
        }

        in_offset += pixels * 2;
    }

    out_buf.truncate(out_p);
    out_buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    #[cfg(feature = "std")]
    #[test]
    fn pure_log_table_matches_std_across_all_halves() {
        let table = b44_log_table();
        for x in 0..=u16::MAX {
            assert_eq!(
                convert_to_linear_pure(x),
                table[x as usize],
                "mismatch at half 0x{x:04x}"
            );
        }
    }

    #[test]
    fn extreme_subsampling_does_not_wrap_or_hang() {
        // Previously `data_width = 4, x_sampling = i32::MAX` wrapped `cw` to 0 and
        // `num_lines = 32, y_sampling = 2_147_483_620` wrapped `ch` to -1 (`usize::MAX`),
        // passing `0 * usize::MAX == 0` in the guard and looping `0..usize::MAX` forever!
        let channels = [
            ChannelInfo {
                name: String::from("Y"),
                pixel_type: PixelType::HALF,
                x_sampling: i32::MAX,
                y_sampling: 2_147_483_620,
                p_linear: 0,
            },
            ChannelInfo {
                name: String::from("Z"),
                pixel_type: PixelType::HALF,
                x_sampling: 1,
                y_sampling: 1,
                p_linear: 0,
            },
        ];
        let mut out = vec![0u8; 4 * 32 * 2];
        // Now `cw = 1, ch = 1` for channel Y, so `expected_out` requires `2 + 256` > `out.len()`
        // and immediately returns `false` without hanging!
        assert!(!decompress_b44(&mut out, &[0xfc; 6], 4, 32, &channels));

        // Even with room in `out` and two flat blocks, it decodes 1x1 + 4x4 cleanly in O(1).
        let mut out_ok = vec![0u8; 2 + 4 * 4 * 2];
        let input = [0x3c, 0x00, 0xfc, 0x3c, 0x00, 0xfc];
        assert!(decompress_b44(&mut out_ok, &input, 4, 4, &channels));
    }
}
