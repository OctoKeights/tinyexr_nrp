//! The ZFP compression tinyexr adds to the format: a port of `FindZFPCompressionParam`,
//! `CompressZfp` and `DecompressZfp` of tinyexr v3.4.0 (`tinyexr.h`), over [`zfp_nrp`], the
//! port of what they use of the zfp library v1.0.1 (<https://github.com/LLNL/zfp>).
//!
//! A block of a ZFP image is its floats taken as one plane of four by four blocks for each
//! channel. Each block is stored with the exponent of its largest value, as integers turned
//! by a transform that leaves most of them small, and then one plane of bits at a time, from
//! the top: as many bits as a fixed rate allows, as many planes as a precision says, or down
//! to a tolerance. The rate, the precision or the tolerance is in attributes of the header
//! (`zfpCompressionType`, `zfpCompressionRate`, `zfpCompressionPrecision`,
//! `zfpCompressionTolerance`), which a reader needs.
//!
//! What zfp's reversible mode needs (a tolerance that is not more than nothing below its
//! least exponent) tinyexr's settings do not reach, and it is not ported.

use super::Attribute;
use zfp_nrp::{BLOCK_LEN, BitStream, Settings};

/// `TINYEXR_ZFP_COMPRESSIONTYPE_RATE`: so many bits a value.
pub(crate) const ZFP_COMPRESSIONTYPE_RATE: i32 = 0;

/// `TINYEXR_ZFP_COMPRESSIONTYPE_PRECISION`: so many planes of bits.
pub(crate) const ZFP_COMPRESSIONTYPE_PRECISION: i32 = 1;

/// `TINYEXR_ZFP_COMPRESSIONTYPE_ACCURACY`: an error no larger than a tolerance.
pub(crate) const ZFP_COMPRESSIONTYPE_ACCURACY: i32 = 2;

/// The settings of ZFP compression, as the attributes a header carries them in (which tinyexr
/// reads on both sides: without them, it writes at 2 bits a value, and does not read).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZfpSettings {
    /// So many bits a value (at least 9 a block of 16 are kept).
    Rate(f64),
    /// As many planes of bits as a float has. (tinyexr reads the precision but keeps it where
    /// the rate goes, so that any precision is this one.)
    Precision(i32),
    /// No error larger than the tolerance.
    Accuracy(f64),
}

impl ZfpSettings {
    /// The attributes, to go in a header's `custom_attributes`.
    pub fn attributes(&self) -> Vec<Attribute> {
        let attribute = |name: &str, type_name: &str, value: Vec<u8>| Attribute {
            name: name.to_string(),
            type_name: type_name.to_string(),
            value,
        };
        let (kind, value) = match *self {
            ZfpSettings::Rate(rate) => (
                ZFP_COMPRESSIONTYPE_RATE,
                attribute("zfpCompressionRate", "double", rate.to_le_bytes().to_vec()),
            ),
            ZfpSettings::Precision(precision) => (
                ZFP_COMPRESSIONTYPE_PRECISION,
                attribute(
                    "zfpCompressionPrecision",
                    "int",
                    precision.to_le_bytes().to_vec(),
                ),
            ),
            ZfpSettings::Accuracy(tolerance) => (
                ZFP_COMPRESSIONTYPE_ACCURACY,
                attribute(
                    "zfpCompressionTolerance",
                    "double",
                    tolerance.to_le_bytes().to_vec(),
                ),
            ),
        };
        vec![
            attribute("zfpCompressionType", "uchar", vec![kind as u8]),
            value,
        ]
    }
}

/// `ZFPCompressionParam`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ZfpParam {
    rate: f64,
    precision: u32,
    tolerance: f64,
    kind: i32,
}

impl Default for ZfpParam {
    fn default() -> Self {
        ZfpParam {
            rate: 2.0,
            precision: 0,
            tolerance: 0.0,
            kind: ZFP_COMPRESSIONTYPE_RATE,
        }
    }
}

/// `FindZFPCompressionParam`: the settings in `attributes`, or none, with why in `err`.
///
/// (As in the original, the precision of `zfpCompressionPrecision` is put where the rate goes,
/// and so a precision is always that of 64 planes: as many as a float has.)
pub(crate) fn find_zfp_compression_param(
    attributes: &[Attribute],
    err: &mut String,
) -> Option<ZfpParam> {
    let mut param = ZfpParam::default();
    let mut found_type = false;

    for attribute in attributes {
        if attribute.name == "zfpCompressionType" {
            if attribute.value.len() == 1 {
                param.kind = i32::from(attribute.value[0]);
                found_type = true;
                break;
            } else {
                err.push_str("zfpCompressionType attribute must be uchar(1 byte) type.\n");
                return None;
            }
        }
    }

    if !found_type {
        err.push_str("`zfpCompressionType` attribute not found.\n");
        return None;
    }

    let value_of = |name: &str, size: usize| {
        attributes
            .iter()
            .find(|attribute| attribute.name == name && attribute.value.len() == size)
            .map(|attribute| attribute.value.as_slice())
    };
    if param.kind == ZFP_COMPRESSIONTYPE_RATE {
        if let Some(value) = value_of("zfpCompressionRate", 8) {
            param.rate = f64::from_le_bytes(value.try_into().expect("eight bytes"));
            return Some(param);
        }
        err.push_str("`zfpCompressionRate` attribute not found.\n");
    } else if param.kind == ZFP_COMPRESSIONTYPE_PRECISION {
        if let Some(value) = value_of("zfpCompressionPrecision", 4) {
            param.rate = f64::from(i32::from_le_bytes(value.try_into().expect("four bytes")));
            return Some(param);
        }
        err.push_str("`zfpCompressionPrecision` attribute not found.\n");
    } else if param.kind == ZFP_COMPRESSIONTYPE_ACCURACY {
        if let Some(value) = value_of("zfpCompressionTolerance", 8) {
            param.tolerance = f64::from_le_bytes(value.try_into().expect("eight bytes"));
            return Some(param);
        }
        err.push_str("`zfpCompressionTolerance` attribute not found.\n");
    } else {
        err.push_str("Unknown value specified for `zfpCompressionType`.\n");
    }

    None
}

/// The settings tinyexr writes with where the header has none of its own (or not all of them).
pub(crate) fn zfp_param_for_writing(attributes: &[Attribute]) -> ZfpParam {
    find_zfp_compression_param(attributes, &mut String::new()).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// zfp

/// The settings of a stream as tinyexr makes them, or none for a type of settings it does not
/// know.
fn zfp_stream(param: &ZfpParam) -> Option<Settings> {
    if param.kind == ZFP_COMPRESSIONTYPE_RATE {
        Some(Settings::fixed_rate(param.rate))
    } else if param.kind == ZFP_COMPRESSIONTYPE_PRECISION {
        Some(Settings::fixed_precision(param.precision))
    } else if param.kind == ZFP_COMPRESSIONTYPE_ACCURACY {
        Some(Settings::fixed_accuracy(param.tolerance))
    } else {
        None
    }
}

/// The most the port gives a stream (tinyexr allocates whatever the settings say).
const MAX_STREAM: usize = 1 << 30;

/// The bytes the stream of a block of `width` by `lines` pixels of `num_channels` channels is
/// given, or none where that is more than a gigabyte.
pub(crate) fn zfp_stream_size(
    param: &ZfpParam,
    width: i32,
    lines: i32,
    num_channels: usize,
) -> Option<usize> {
    let zfp = zfp_stream(param)?;
    let size = zfp.max_stream_len(
        width.max(0) as usize,
        (lines as u32).wrapping_mul(num_channels as u32) as usize,
    );
    (size <= MAX_STREAM).then_some(size)
}

/// Why `compress_zfp` does not compress a block.
pub(crate) enum ZfpRefusal {
    /// Its width or its lines are not a multiple of four, which the original does not look at
    /// past its own refusal (it then writes a length it never set).
    NotMultipleOfFour,
    /// Its settings ask for more than a gigabyte.
    TooLarge,
}

/// `CompressZfp`: `num_lines` lines of `width` floats of each of `num_channels` channels, one
/// plane after the other (as tinyexr takes the block it hands over, whose lines are in fact
/// the channels of one line after the other), compressed.
pub(crate) fn compress_zfp(
    input: &[u8],
    width: i32,
    num_lines: i32,
    num_channels: usize,
    param: &ZfpParam,
) -> Result<Vec<u8>, ZfpRefusal> {
    if width % 4 != 0 || num_lines % 4 != 0 {
        return Err(ZfpRefusal::NotMultipleOfFour);
    }
    // (`with` refuses only a type of settings that `zfp_param_for_writing` does not give.)
    let zfp = zfp_stream(param).ok_or(ZfpRefusal::NotMultipleOfFour)?;
    let (width, num_lines) = (width as usize, num_lines as usize);

    let buf_size = zfp.max_stream_len(
        width,
        (num_lines as u32).wrapping_mul(num_channels as u32) as usize,
    );
    if buf_size > MAX_STREAM {
        return Err(ZfpRefusal::TooLarge);
    }
    let mut stream = BitStream::new(vec![0u64; buf_size / 8]);

    let image_size = width * num_lines;
    let float_at = |index: usize| {
        f32::from_bits(u32::from_le_bytes(
            input[4 * index..4 * index + 4]
                .try_into()
                .expect("four bytes"),
        ))
    };
    for c in 0..num_channels {
        // compress 4x4 pixel block.
        for y in (0..num_lines).step_by(4) {
            for x in (0..width).step_by(4) {
                let mut fblock = [0f32; BLOCK_LEN];
                for j in 0..4 {
                    for i in 0..4 {
                        fblock[j * 4 + i] = float_at(c * image_size + ((y + j) * width + (x + i)));
                    }
                }
                zfp.encode_block(&mut stream, &fblock);
            }
        }
    }

    stream.flush();
    Ok(stream.to_bytes())
}

/// `DecompressZfp`: `src` into `dst`, `dst_num_lines` lines of `dst_width` floats of each of
/// `num_channels` channels, one plane after the other. The answer, which the original's
/// caller does not look at, is whether it was done; none where the settings ask for more
/// than a gigabyte, which stops the original where it can have no more.
///
/// The original takes a block of as many bytes as the block has values for one that is not
/// compressed, copies it, and then goes on; and writes past `dst` for a block with a channel
/// that is not of floats (which then is not read), and past the buffer it reads from for a
/// block longer than the most its settings make. Here what is past them is left out.
pub(crate) fn decompress_zfp(
    dst: &mut [u8],
    dst_width: i32,
    dst_num_lines: i32,
    num_channels: usize,
    src: &[u8],
    param: &ZfpParam,
) -> Option<bool> {
    let uncompressed_size = (dst_width as usize)
        .wrapping_mul(dst_num_lines as usize)
        .wrapping_mul(num_channels);

    if uncompressed_size == src.len() {
        // Data is not compressed(Issue 40).
        dst[..src.len()].copy_from_slice(src);
    }

    if dst_width % 4 != 0 || dst_num_lines % 4 != 0 {
        return Some(false);
    }

    let Some(zfp) = zfp_stream(param) else {
        return Some(false);
    };
    let (width, num_lines) = (dst_width as usize, dst_num_lines as usize);

    let buf_size = zfp.max_stream_len(
        width,
        (num_lines as u32).wrapping_mul(num_channels as u32) as usize,
    );
    if buf_size > MAX_STREAM {
        return None;
    }
    // (What is past the most the settings make is left out.)
    let mut stream = BitStream::from_bytes(&src[..src.len().min(buf_size)], buf_size / 8);

    let image_size = width * num_lines;
    for c in 0..num_channels {
        // decompress 4x4 pixel block.
        for y in (0..num_lines).step_by(4) {
            for x in (0..width).step_by(4) {
                let mut fblock = [0f32; BLOCK_LEN];
                zfp.decode_block(&mut stream, &mut fblock);
                for j in 0..4 {
                    for i in 0..4 {
                        let at = 4 * (c * image_size + ((y + j) * width + (x + i)));
                        if let Some(slot) = dst.get_mut(at..at + 4) {
                            slot.copy_from_slice(&fblock[j * 4 + i].to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
    }

    Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_come_back() {
        // At full precision, smooth values come back nearly as they were.
        let param = ZfpParam {
            kind: ZFP_COMPRESSIONTYPE_PRECISION,
            ..ZfpParam::default()
        };
        let (width, lines, channels) = (8, 4, 2);
        let mut input = Vec::new();
        let mut values = Vec::new();
        for k in 0..width * lines * channels {
            let value = (k as f32 * 0.37).sin() * 10.0;
            values.push(value);
            input.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let Ok(packed) = compress_zfp(&input, width as i32, lines as i32, channels, &param) else {
            panic!("not compressed")
        };
        let mut out = vec![0u8; input.len()];
        assert_eq!(
            decompress_zfp(
                &mut out,
                width as i32,
                lines as i32,
                channels,
                &packed,
                &param
            ),
            Some(true)
        );
        for (k, &value) in values.iter().enumerate() {
            let got = f32::from_bits(u32::from_le_bytes(
                out[4 * k..4 * k + 4].try_into().unwrap(),
            ));
            assert!((got - value).abs() < 1e-4, "{k}: {got} {value}");
        }
    }
}
