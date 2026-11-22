//! The ZFP compression tinyexr adds to the format: a port of `FindZFPCompressionParam`,
//! `CompressZfp` and `DecompressZfp` of tinyexr v3.4.0 (`tinyexr.h`), over `zfp_nrp`, the
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

use alloc::{string::ToString, vec, vec::Vec};

use super::Attribute;
#[cfg(feature = "zfp")]
use zfp_nrp::{BLOCK_LEN, BitReader, BitStream, Settings};

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

impl Default for ZfpSettings {
    /// tinyexr's default ZFP setting: fixed rate of 2 bits per value (`Rate(2.0)`).
    fn default() -> Self {
        ZfpSettings::Rate(2.0)
    }
}

impl ZfpSettings {
    /// The attributes, to go in a header's `custom_attributes`.
    #[must_use]
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
#[allow(dead_code)]
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

/// `FindZFPCompressionParam`: the settings in `attributes`, or why there are none.
///
/// (As in the original, the precision of `zfpCompressionPrecision` is put where the rate goes,
/// and so a precision is always that of 64 planes: as many as a float has.)
#[allow(dead_code)]
pub(crate) fn find_zfp_compression_param(
    attributes: &[Attribute],
) -> Result<ZfpParam, &'static str> {
    let Some(kind) = attributes
        .iter()
        .find(|attribute| attribute.name == "zfpCompressionType")
    else {
        return Err("`zfpCompressionType` attribute not found.\n");
    };
    let [kind] = kind.value[..] else {
        return Err("zfpCompressionType attribute must be uchar(1 byte) type.\n");
    };
    let mut param = ZfpParam {
        kind: i32::from(kind),
        ..ZfpParam::default()
    };
    match param.kind {
        ZFP_COMPRESSIONTYPE_RATE => {
            let value = value_of(attributes, "zfpCompressionRate")
                .ok_or("`zfpCompressionRate` attribute not found.\n")?;
            param.rate = f64::from_le_bytes(value);
        }
        ZFP_COMPRESSIONTYPE_PRECISION => {
            let value = value_of(attributes, "zfpCompressionPrecision")
                .ok_or("`zfpCompressionPrecision` attribute not found.\n")?;
            param.rate = f64::from(i32::from_le_bytes(value));
        }
        ZFP_COMPRESSIONTYPE_ACCURACY => {
            let value = value_of(attributes, "zfpCompressionTolerance")
                .ok_or("`zfpCompressionTolerance` attribute not found.\n")?;
            param.tolerance = f64::from_le_bytes(value);
        }
        _ => return Err("Unknown value specified for `zfpCompressionType`.\n"),
    }
    Ok(param)
}

/// The value of the first attribute named `name` that is `N` bytes long.
#[allow(dead_code)]
fn value_of<const N: usize>(attributes: &[Attribute], name: &str) -> Option<[u8; N]> {
    attributes
        .iter()
        .filter(|attribute| attribute.name == name)
        .find_map(|attribute| attribute.value[..].try_into().ok())
}

/// The settings tinyexr writes with where the header has none of its own (or not all of them).
#[allow(dead_code)]
pub(crate) fn zfp_param_for_writing(attributes: &[Attribute]) -> ZfpParam {
    find_zfp_compression_param(attributes).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// zfp

/// The settings of a stream as tinyexr makes them, or none for a type of settings it does not
/// know.
#[cfg(feature = "zfp")]
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
#[cfg(feature = "zfp")]
const MAX_STREAM: usize = 1 << 30;

/// The bytes the stream of a block of `width` by `lines` pixels of `num_channels` channels is
/// given, or none where that is more than a gigabyte or overflows `usize`.
#[cfg(feature = "zfp")]
pub(crate) fn zfp_stream_size(
    param: &ZfpParam,
    width: i32,
    lines: i32,
    num_channels: usize,
) -> Option<usize> {
    if width < 0 || lines < 0 {
        return None;
    }
    let zfp = zfp_stream(param)?;
    let height = (lines as usize).checked_mul(num_channels)?;
    let size = zfp.checked_max_stream_len(width as usize, height)?;
    (size <= MAX_STREAM).then_some(size)
}

/// Why `compress_zfp` does not compress a block.
#[cfg(feature = "zfp")]
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
#[cfg(feature = "zfp")]
pub(crate) fn compress_zfp(
    input: &[u8],
    width: i32,
    num_lines: i32,
    num_channels: usize,
    param: &ZfpParam,
) -> Result<Vec<u8>, ZfpRefusal> {
    if width < 0 || num_lines < 0 || width % 4 != 0 || num_lines % 4 != 0 {
        return Err(ZfpRefusal::NotMultipleOfFour);
    }
    let zfp = zfp_stream(param).ok_or(ZfpRefusal::NotMultipleOfFour)?;
    let buf_size =
        zfp_stream_size(param, width, num_lines, num_channels).ok_or(ZfpRefusal::TooLarge)?;
    let (width, num_lines) = (width as usize, num_lines as usize);

    // Reserve only what the input block typically needs rather than zero-filling up to 1 GiB;
    // `BitStream` grows dynamically if a pathological setting writes more bits than raw.
    let initial_words = (buf_size / 8).min(input.len() / 8 + 16);
    let mut stream = BitStream::with_capacity(initial_words);

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
/// `num_channels` channels, one plane after the other, using a zero-copy [`BitReader`].
#[cfg(feature = "zfp")]
pub(crate) fn decompress_zfp(
    dst: &mut [u8],
    dst_width: i32,
    dst_num_lines: i32,
    num_channels: usize,
    src: &[u8],
    param: &ZfpParam,
) -> Option<bool> {
    if dst_width < 0 || dst_num_lines < 0 {
        return Some(false);
    }
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
    let buf_size = zfp_stream_size(param, dst_width, dst_num_lines, num_channels)?;
    let (width, num_lines) = (dst_width as usize, dst_num_lines as usize);

    // Zero-copy BitReader over the bounded stream slice (no 1 GiB word buffer allocation!).
    let mut reader = BitReader::new(&src[..src.len().min(buf_size)]);

    let image_size = width * num_lines;
    for c in 0..num_channels {
        // decompress 4x4 pixel block.
        for y in (0..num_lines).step_by(4) {
            for x in (0..width).step_by(4) {
                let mut fblock = [0f32; BLOCK_LEN];
                zfp.decode_block_slice(&mut reader, &mut fblock);
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

#[cfg(all(test, feature = "zfp"))]
mod tests {
    use super::*;

    #[test]
    fn blocks_come_back_and_default_is_rate_2() {
        assert_eq!(ZfpSettings::default(), ZfpSettings::Rate(2.0));
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
