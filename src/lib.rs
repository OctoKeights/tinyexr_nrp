//! `.exr` files (OpenEXR) read and written in pure, safe Rust: a native port of Syoyo Fujita's
//! [tinyexr](https://github.com/syoyo/tinyexr) v3.4.0, which writes the files tinyexr writes,
//! byte for byte, reads the values it reads, and refuses the files it refuses, with the same
//! code and message ([`ExrError`]).
//!
//! - [`load_rgba`] reads a file as red, green, blue and alpha floats, and [`save_floats`]
//!   writes floats of one, three or four channels: what most callers want.
//!
//! - [`parse_version`], [`parse_header`] and [`load_image`] give the header and then every
//!   channel as it is stored, or every tile of every level; [`save_image`] writes an image
//!   the same way, with [`tile_levels`] for the levels a tiled one has.
//!
//! - [`parse_multipart_headers`], [`load_multipart_images`], [`load_multipart_part`] and
//!   [`save_multipart_images`] do the same for files of several parts, and [`load_deep`] and
//!   [`save_deep`] read and write deep images.
//!
//! - With the default `std` feature, [`load_image_parallel`], [`load_rgba_parallel`],
//!   [`save_image_parallel`] and [`save_floats_parallel`] compress and decompress chunks across
//!   threads (producing byte-for-byte identical files), alongside [`load_rgba_file`],
//!   [`save_floats_file`], [`read_from`] and [`write_to`].
//!
//! Every compression tinyexr has is supported: RLE, ZIPS, ZIP, PIZ, PXR24, B44, B44A, and
//! ZFP for floats ([`ZfpSettings`], with the default `zfp` feature). DWAA and DWAB, which
//! tinyexr refuses, are read and written as OpenEXR 3.5 does (with the default `dwa` feature,
//! at the level of [`dwa_compression_level_attribute`]).
//!
//! Where tinyexr departs from the format, the functions ending in `_as_tinyexr` follow it
//! exactly; the others follow the format. Damaged files that would have tinyexr read past
//! their end, or stop, are refused. The
//! [Guide](https://github.com/OctoKeights/tinyexr_nrp/blob/main/docs/guide.md) has the details.
//!
//! ```
//! use tinyexr_nrp::{load_rgba, save_floats};
//!
//! let rgba = vec![0.5f32; 16 * 16 * 4];
//! let file = save_floats(&rgba, 16, 16, 4, true).unwrap();
//! let image = load_rgba(&file).unwrap();
//! assert_eq!((image.width, image.height), (16, 16));
//! assert_eq!(image.data, rgba);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

mod b44;
mod chunk;
mod decode;
mod deep;
mod encode;
mod parts;
mod piz;
mod pxr24;
mod zfp;
mod zip_rle;

pub use chunk::{TileLevel, tile_levels};
pub use decode::ChannelData;
pub use deep::{DeepImage, load_deep, save_deep};
pub use encode::{
    DWA_INVALID_LEVEL, ZFP_NOT_MULTIPLE_OF_FOUR, ZFP_TOO_LARGE, dwa_compression_level_attribute,
    save_floats, save_image, save_image_as_tinyexr, save_multipart_images,
    save_multipart_images_as_tinyexr,
};
#[cfg(feature = "std")]
pub use encode::{save_floats_parallel, save_image_parallel};
pub use parts::{
    PAST_THE_END, load_multipart_images, load_multipart_images_as_tinyexr, load_multipart_part,
    parse_multipart_headers, parse_multipart_headers_as_tinyexr,
};
pub use zfp::ZfpSettings;

const MAX_HEADER_ATTRIBUTES: usize = 1024;
const MAX_CUSTOM_ATTRIBUTES: usize = 128;

/// Maximum byte length of OpenEXR attribute, channel, and part names kept by tinyexr.
const MAX_NAME_LEN: usize = 255;

/// Defines a code a file or tinyexr numbers something by: a newtype of the number, which
/// keeps any value a file holds, with constants for those that have a meaning.
macro_rules! code {
    ($(#[$doc:meta])* $name:ident { $($(#[$cdoc:meta])* $constant:ident = $value:expr),* $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub i32);

        impl $name {
            $($(#[$cdoc])* pub const $constant: $name = $name($value);)*

            /// The name of the constant with this value, if there is one.
            #[must_use]
            pub const fn name(self) -> Option<&'static str> {
                match self.0 {
                    $($value => Some(stringify!($constant)),)*
                    _ => None,
                }
            }
        }

        impl From<i32> for $name {
            fn from(value: i32) -> Self {
                $name(value)
            }
        }

        impl From<$name> for i32 {
            fn from(code: $name) -> i32 {
                code.0
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self.name() {
                    Some(name) => write!(f, "{}::{}", stringify!($name), name),
                    None => write!(f, "{}({})", stringify!($name), self.0),
                }
            }
        }
    };
}

code! {
    /// The order of a file's blocks: the `lineOrder` attribute, which keeps any value a file
    /// holds. Only `INCREASING_Y` reads lines top first; tinyexr reads every other bottom first.
    LineOrder {
        /// Lines from the top.
        INCREASING_Y = 0,
        /// Lines from the bottom.
        DECREASING_Y = 1,
        /// Blocks in any order.
        RANDOM_Y = 2,
    }
}

code! {
    /// One of the original's error codes (`TINYEXR_ERROR_*`).
    ErrorCode {
        /// No error (`TINYEXR_SUCCESS`), which the original's functions return when they
        /// succeed.
        SUCCESS = 0,
        /// The file does not start with OpenEXR's magic number.
        INVALID_MAGIC_NUMBER = -1,
        /// The file is not of version 2 of the format.
        INVALID_EXR_VERSION = -2,
        /// An argument is invalid.
        INVALID_ARGUMENT = -3,
        /// The file's data is invalid.
        INVALID_DATA = -4,
        /// The file is invalid.
        INVALID_FILE = -5,
        /// A parameter is invalid.
        INVALID_PARAMETER = -6,
        /// The file is of a kind that is not supported.
        UNSUPPORTED_FORMAT = -8,
        /// The header is invalid.
        INVALID_HEADER = -9,
        /// The file uses a feature that is not supported.
        UNSUPPORTED_FEATURE = -10,
        /// The image could not be written.
        SERIALIZATION_FAILED = -12,
        /// A layer asked for is not in the file.
        LAYER_NOT_FOUND = -13,
        /// The image is too large.
        DATA_TOO_LARGE = -14,
    }
}

impl core::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::SUCCESS => f.write_str("no error"),
            Self::INVALID_MAGIC_NUMBER => f.write_str("invalid OpenEXR magic number"),
            Self::INVALID_EXR_VERSION => f.write_str("unsupported OpenEXR version"),
            Self::INVALID_ARGUMENT => f.write_str("invalid argument"),
            Self::INVALID_DATA => f.write_str("invalid EXR data"),
            Self::INVALID_FILE => f.write_str("invalid EXR file"),
            Self::INVALID_PARAMETER => f.write_str("invalid EXR parameter"),
            Self::UNSUPPORTED_FORMAT => f.write_str("unsupported EXR format"),
            Self::INVALID_HEADER => f.write_str("invalid EXR header"),
            Self::UNSUPPORTED_FEATURE => f.write_str("unsupported EXR feature"),
            Self::SERIALIZATION_FAILED => f.write_str("EXR serialization failed"),
            Self::LAYER_NOT_FOUND => f.write_str("EXR layer not found"),
            Self::DATA_TOO_LARGE => f.write_str("EXR data too large"),
            Self(code) => write!(f, "tinyexr error code {code}"),
        }
    }
}

impl core::error::Error for ErrorCode {}

code! {
    /// The type of a channel's values (`TINYEXR_PIXELTYPE_*`).
    PixelType {
        /// 32-bit unsigned integers.
        UINT = 0,
        /// 16-bit floats, halves.
        HALF = 1,
        /// 32-bit floats.
        FLOAT = 2,
    }
}

code! {
    /// How the pixels of a file are compressed (`TINYEXR_COMPRESSIONTYPE_*`).
    Compression {
        /// Stored as they are.
        NONE = 0,
        /// Run-length encoded.
        RLE = 1,
        /// zlib, a line to a block.
        ZIPS = 2,
        /// zlib, 16 lines to a block.
        ZIP = 3,
        /// Wavelet and Huffman coded, 32 lines to a block.
        PIZ = 4,
        /// Lossy for floats (rounded to 24 bits), then zlib.
        PXR24 = 5,
        /// Lossy, halves in blocks of four by four.
        B44 = 6,
        /// B44, with flat blocks smaller.
        B44A = 7,
        /// DreamWorks Animation's lossy compression, 32 lines to a block, which tinyexr does
        /// not have (the functions that end in `_as_tinyexr` refuse it as tinyexr does). The
        /// `dwaCompressionLevel` attribute of a header, which
        /// [`dwa_compression_level_attribute`] makes, says how lossy (45 without it).
        DWAA = 8,
        /// The same, 256 lines to a block.
        DWAB = 9,
        /// A tinyexr extension, of floats only, with the zfp library's coder; its settings
        /// are in attributes of the header, which [`ZfpSettings`] makes.
        ZFP = 128,
    }
}

code! {
    /// The levels of a tiled file (`TINYEXR_TILE_*_LEVEL(S)`).
    TileLevelMode {
        /// One level.
        ONE_LEVEL = 0,
        /// Mipmap levels: each half the size of the one before both ways.
        MIPMAP_LEVELS = 1,
        /// Ripmap levels: each half the size of another one way or the other.
        RIPMAP_LEVELS = 2,
    }
}

code! {
    /// How the sizes of a tiled file's levels are rounded (`TINYEXR_TILE_ROUND_*`).
    TileRoundingMode {
        /// Rounded down.
        ROUND_DOWN = 0,
        /// Rounded up.
        ROUND_UP = 1,
    }
}

/// Why a file was refused: the original's error code and the message it gives with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExrError {
    /// The original's error code.
    pub code: ErrorCode,
    /// The message the original gives with it, often several lines, each ended by a line
    /// break.
    pub message: String,
}

impl core::fmt::Display for ExrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.message.is_empty() {
            write!(
                f,
                "the .exr file could not be read or written (tinyexr's error {})",
                self.code.0
            )
        } else {
            f.write_str(self.message.trim_end())
        }
    }
}

impl core::error::Error for ExrError {}

impl ExrError {
    /// An error of `code`, with `message`.
    pub(crate) fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        ExrError {
            code,
            message: message.into(),
        }
    }
}

/// An error of the code, with no message (as the original gives some of its codes).
impl From<ErrorCode> for ExrError {
    fn from(code: ErrorCode) -> Self {
        ExrError::new(code, String::new())
    }
}

/// `EXRVersion`: the eight bytes a file starts with.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExrVersion {
    /// This will be 2.
    pub version: i32,
    /// Tile format image; not zero only for a single-part "normal" image (not deep, not
    /// multipart).
    pub tiled: bool,
    /// Long name attribute.
    pub long_name: bool,
    /// Deep image (EXR 2.0); zero for an OpenEXR 1.x image.
    pub non_image: bool,
    /// Multi-part (EXR 2.0); zero for an OpenEXR 1.x image.
    pub multipart: bool,
}

/// `EXRAttribute`: an attribute of the header that the reader has no field for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Attribute {
    /// Its name.
    pub name: String,
    /// The name of its type, such as `"string"` or `"box2i"`.
    pub type_name: String,
    /// Its value as the file has it.
    pub value: Vec<u8>,
}

/// `EXRChannelInfo`: a channel of a file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChannelInfo {
    /// Its name, such as `"R"`.
    pub name: String,
    /// The type of its values in the file.
    pub pixel_type: PixelType,
    /// Its horizontal sampling: 1 for a value at every pixel.
    pub x_sampling: i32,
    /// Its vertical sampling: 1 for a value on every line.
    pub y_sampling: i32,
    /// 1 if its values are perceptually linear (which DWAA and DWAB take into account).
    pub p_linear: u8,
}

/// `EXRBox2i`: a rectangle of pixels, its corners included.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Box2i {
    /// The left edge.
    pub min_x: i32,
    /// The top edge.
    pub min_y: i32,
    /// The right edge (inclusive).
    pub max_x: i32,
    /// The bottom edge (inclusive).
    pub max_y: i32,
}

/// The original's `HeaderInfo`: a header as it is parsed, before it becomes an `ExrHeader`.
#[derive(Clone, Debug, Default)]
pub(crate) struct HeaderInfo {
    channels: Vec<ChannelInfo>,
    attributes: Vec<Attribute>,

    data_window: Box2i,
    line_order: LineOrder,
    display_window: Box2i,
    screen_window_center: [f32; 2],
    screen_window_width: f32,
    pixel_aspect_ratio: f32,

    chunk_count: i32,

    // Tiled format
    tiled: bool,
    tile_size_x: i32,
    tile_size_y: i32,
    tile_level_mode: TileLevelMode,
    tile_rounding_mode: TileRoundingMode,

    header_len: u32,

    compression_type: Compression,

    // required for multi-part or non-image files
    name: String,
    // required for multi-part or non-image files
    type_name: String,
}

/// The original's `FP16` and `half_to_float`: a half as a float, by shifting its bits into
/// place and, for the smallest values, subtracting where a float's hidden bit would be.
#[must_use]
pub fn half_to_float(h: u16) -> f32 {
    const SHIFTED_EXP: u32 = 0x7c00 << 13;
    let h = u32::from(h);

    let mut o = (h & 0x7fff) << 13;
    let exp_ = SHIFTED_EXP & o;
    o = o.wrapping_add((127 - 15) << 23);

    if exp_ == SHIFTED_EXP {
        o = o.wrapping_add((128 - 16) << 23);
    } else if exp_ == 0 {
        let magic = f32::from_bits(113 << 23);
        o = o.wrapping_add(1 << 23);
        o = (f32::from_bits(o) - magic).to_bits();
    }

    o |= (h & 0x8000) << 16;
    f32::from_bits(o)
}

/// The original's `float_to_half`: a float as a half, rounded to the nearest with a
/// half going up.
#[must_use]
pub fn float_to_half(f: f32) -> u16 {
    let bits = f.to_bits();
    let f_mantissa = bits & 0x007f_ffff;
    let f_exponent = (bits >> 23) & 0xff;
    let f_sign = bits >> 31;
    let mut o: u16 = 0;
    let set = |o: &mut u16, exponent: u32, mantissa: u32| {
        *o = (*o & 0x8000) | (((exponent & 0x1f) as u16) << 10) | (mantissa & 0x3ff) as u16;
    };

    if f_exponent == 0 {
        set(&mut o, 0, 0);
    } else if f_exponent == 255 {
        set(&mut o, 31, if f_mantissa != 0 { 0x200 } else { 0 });
    } else {
        let newexp = f_exponent as i32 - 127 + 15;
        if newexp >= 31 {
            set(&mut o, 31, 0);
        } else if newexp <= 0 {
            if (14 - newexp) <= 24 {
                let mant = f_mantissa | 0x0080_0000;
                set(&mut o, 0, mant >> (14 - newexp));
                if (mant >> (13 - newexp)) & 1 != 0 {
                    o = o.wrapping_add(1);
                }
            }
        } else {
            set(&mut o, newexp as u32, f_mantissa >> 13);
            if f_mantissa & 0x1000 != 0 {
                o = o.wrapping_add(1);
            }
        }
    }

    (o & 0x7fff) | ((f_sign as u16) << 15)
}

/// Truncates `bytes` to at most `MAX_NAME_LEN` (`255`) raw bytes *before* lossy UTF-8
/// conversion, preventing megabyte allocations and mid-replacement UTF-8 splits.
#[inline]
fn bytes_to_name_string(bytes: &[u8]) -> String {
    let bounded = &bytes[..bytes.len().min(MAX_NAME_LEN)];
    String::from_utf8_lossy(bounded).into_owned()
}

/// `ReadString`: a string that ends in a zero within the `len` bytes at `ptr` of `data`.
/// The answer is the string (up to 255 raw bytes) and where the byte after its zero is, or
/// `None` when there is no zero.
fn read_string(data: &[u8], ptr: usize, len: usize) -> Option<(String, usize)> {
    let mut q = ptr;
    while q - ptr < len && data[q] != 0 {
        q += 1;
    }

    if q - ptr >= len {
        return None;
    }

    Some((bytes_to_name_string(&data[ptr..q]), q + 1))
}

/// `ReadAttribute`: the attribute at the start of `marker`: its name, the name of its type,
/// its value, and how many bytes all of it took.
pub(crate) fn read_attribute(marker: &[u8]) -> Option<(String, String, Vec<u8>, usize)> {
    let mut size = marker.len();
    let mut at = 0usize;
    let strnlen = |from: usize, size: usize| {
        marker[from..from + size]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(size)
    };

    let name_len = strnlen(at, size);
    if name_len == size {
        return None;
    }
    let name = bytes_to_name_string(&marker[at..at + name_len]);

    at += name_len + 1;
    size -= name_len + 1;

    let type_len = strnlen(at, size);
    if type_len == size {
        return None;
    }
    let type_name = bytes_to_name_string(&marker[at..at + type_len]);

    at += type_len + 1;
    size -= type_len + 1;

    if size < 4 {
        return None;
    }

    let data_len =
        u32::from_le_bytes([marker[at], marker[at + 1], marker[at + 2], marker[at + 3]]) as usize;

    if data_len == 0 {
        if type_name == "string" {
            return Some((name, type_name, vec![0], name_len + 1 + type_len + 1 + 4));
        }
        return None;
    }

    at += 4;
    size -= 4;

    if size < data_len {
        return None;
    }

    let data = marker[at..at + data_len].to_vec();

    Some((
        name,
        type_name,
        data,
        name_len + 1 + type_len + 1 + 4 + data_len,
    ))
}

/// `ReadChannelInfo`: the channels of a `channels` attribute, added to `channels`.
pub(crate) fn read_channel_info(channels: &mut Vec<ChannelInfo>, data: &[u8]) -> bool {
    let mut p = 0usize;

    loop {
        if p >= data.len() {
            return false;
        }
        if data[p] == 0 {
            break;
        }
        let data_len = data.len() - p;

        let Some((name, after)) = read_string(data, p, data_len) else {
            return false;
        };
        p = after;

        let Some(data_end) = p.checked_add(16) else {
            return false;
        };
        if data_end >= data.len() {
            return false;
        }

        let int_at =
            |at: usize| i32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        let pixel_type = int_at(p);
        p += 4;
        let p_linear = data[p];
        p += 1 + 3; // reserved: uchar[3]
        let x_sampling = int_at(p);
        p += 4;
        let y_sampling = int_at(p);
        p += 4;

        channels.push(ChannelInfo {
            name,
            pixel_type: PixelType(pixel_type),
            x_sampling,
            y_sampling,
            p_linear,
        });
    }

    true
}

/// What [`parse_exr_header`] found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Parsed {
    /// A header.
    Header,
    /// The empty header that ends the list of headers in a file of many parts.
    EndOfHeaders,
}

/// `ParseEXRHeader`: the attributes of a header at the start of `buf`, into `info`.
#[allow(clippy::too_many_lines)]
pub(crate) fn parse_exr_header(
    info: &mut HeaderInfo,
    version: ExrVersion,
    buf: &[u8],
    as_tinyexr: bool,
) -> Result<Parsed, ExrError> {
    let mut size = buf.len();
    let mut marker = 0usize;

    if version.multipart && size > 0 && buf[marker] == 0 {
        return Ok(Parsed::EndOfHeaders);
    }

    let mut has_channels = false;
    let mut has_compression = false;
    let mut has_data_window = false;
    let mut has_display_window = false;
    let mut has_line_order = false;
    let mut has_pixel_aspect_ratio = false;
    let mut has_screen_window_center = false;
    let mut has_screen_window_width = false;
    let mut has_name = false;
    let mut has_type = false;

    info.name.clear();
    info.type_name.clear();

    info.data_window = Box2i::default();
    info.line_order = LineOrder::INCREASING_Y;
    info.display_window = Box2i::default();
    info.screen_window_center = [0.0, 0.0];
    info.screen_window_width = -1.0;
    info.pixel_aspect_ratio = -1.0;

    info.tiled = false;
    info.tile_size_x = -1;
    info.tile_size_y = -1;
    info.tile_level_mode = TileLevelMode(-1);
    info.tile_rounding_mode = TileRoundingMode(-1);

    info.attributes.clear();

    let orig_size = size;
    for _nattr in 0..MAX_HEADER_ATTRIBUTES {
        if 0 == size {
            return Err(ExrError::new(
                ErrorCode::INVALID_DATA,
                "Insufficient data size for attributes.\n",
            ));
        } else if buf[marker] == 0 {
            size -= 1;
            break;
        }

        let Some((attr_name, attr_type, data, marker_size)) =
            read_attribute(&buf[marker..marker + size])
        else {
            return Err(ExrError::new(
                ErrorCode::INVALID_DATA,
                "Failed to read attribute.\n",
            ));
        };
        marker += marker_size;
        size -= marker_size;

        let int_at = |data: &[u8], at: usize| {
            i32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
        };
        let float_at = |data: &[u8], at: usize| f32::from_bits(int_at(data, at) as u32);
        let box_at = |data: &[u8]| Box2i {
            min_x: int_at(data, 0),
            min_y: int_at(data, 4),
            max_x: int_at(data, 8),
            max_y: int_at(data, 12),
        };

        if (version.tiled || version.multipart || version.non_image) && attr_name == "tiles" {
            if data.len() != 9 {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "(ParseEXRHeader) Invalid attribute data size. Attribute data size must be 9.\n",
                ));
            }

            let x_size = int_at(&data, 0) as u32;
            let y_size = int_at(&data, 4) as u32;
            let tile_mode = data[8];

            if x_size > i32::MAX as u32 || y_size > i32::MAX as u32 {
                return Err(ExrError::new(
                    ErrorCode::UNSUPPORTED_FORMAT,
                    "Tile sizes were invalid.",
                ));
            }
            info.tile_size_x = x_size as i32;
            info.tile_size_y = y_size as i32;

            info.tile_level_mode = TileLevelMode(i32::from(tile_mode & 0x3));
            info.tile_rounding_mode = TileRoundingMode(i32::from((tile_mode >> 4) & 0x1));
            info.tiled = true;
        } else if attr_name == "compression" {
            let mut ok = false;
            let compression = Compression(i32::from(data[0]));
            if compression < Compression::PIZ {
                ok = true;
            }

            if compression == Compression::PIZ {
                ok = true;
            }

            #[cfg(feature = "zfp")]
            if compression == Compression::ZFP {
                ok = true;
            }

            if compression == Compression::PXR24
                || compression == Compression::B44
                || compression == Compression::B44A
            {
                ok = true;
            }

            #[cfg(feature = "dwa")]
            if !as_tinyexr && (compression == Compression::DWAA || compression == Compression::DWAB)
            {
                ok = true;
            }
            let _ = as_tinyexr;

            if !ok {
                return Err(ExrError::new(
                    ErrorCode::UNSUPPORTED_FORMAT,
                    "Unknown compression type.",
                ));
            }

            info.compression_type = compression;
            has_compression = true;
        } else if attr_name == "channels" {
            if !read_channel_info(&mut info.channels, &data) {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "Failed to parse channel info.\n",
                ));
            }

            if info.channels.is_empty() {
                return Err(ExrError::new(
                    ErrorCode::INVALID_DATA,
                    "# of channels is zero.\n",
                ));
            }

            has_channels = true;
        } else if attr_name == "dataWindow" {
            if data.len() >= 16 {
                info.data_window = box_at(&data);
                has_data_window = true;
            }
        } else if attr_name == "displayWindow" {
            if data.len() >= 16 {
                info.display_window = box_at(&data);
                has_display_window = true;
            }
        } else if attr_name == "lineOrder" {
            if !data.is_empty() {
                info.line_order = LineOrder(i32::from(data[0]));
                has_line_order = true;
            }
        } else if attr_name == "pixelAspectRatio" {
            if data.len() >= 4 {
                info.pixel_aspect_ratio = float_at(&data, 0);
                has_pixel_aspect_ratio = true;
            }
        } else if attr_name == "screenWindowCenter" {
            if data.len() >= 8 {
                info.screen_window_center = [float_at(&data, 0), float_at(&data, 4)];
                has_screen_window_center = true;
            }
        } else if attr_name == "screenWindowWidth" {
            if data.len() >= 4 {
                info.screen_window_width = float_at(&data, 0);
                has_screen_window_width = true;
            }
        } else if attr_name == "chunkCount" {
            if data.len() >= 4 {
                info.chunk_count = int_at(&data, 0);
            }
        } else if attr_name == "name" {
            if !data.is_empty() && data[0] != 0 {
                let len = data.iter().position(|&b| b == 0).unwrap_or(data.len());
                info.name = bytes_to_name_string(&data[..len]);
                has_name = true;
            }
        } else if attr_name == "type" {
            if !data.is_empty() && data[0] != 0 {
                let len = data.iter().position(|&b| b == 0).unwrap_or(data.len());
                info.type_name = bytes_to_name_string(&data[..len]);
                has_type = true;
            }
        } else if info.attributes.len() < MAX_CUSTOM_ATTRIBUTES {
            info.attributes.push(Attribute {
                name: attr_name,
                type_name: attr_type,
                value: data,
            });
        }
    }

    // Check if required attributes exist.
    {
        let mut missing = String::new();

        if !has_compression {
            missing.push_str("\"compression\" attribute not found in the header.\n");
        }
        if !has_channels {
            missing.push_str("\"channels\" attribute not found in the header.\n");
        }
        if !has_line_order {
            missing.push_str("\"lineOrder\" attribute not found in the header.\n");
        }
        if !has_display_window {
            missing.push_str("\"displayWindow\" attribute not found in the header.\n");
        }
        if !has_data_window {
            missing.push_str("\"dataWindow\" attribute not found in the header or invalid.\n");
        }
        if !has_pixel_aspect_ratio {
            missing.push_str("\"pixelAspectRatio\" attribute not found in the header.\n");
        }
        if !has_screen_window_width {
            missing.push_str("\"screenWindowWidth\" attribute not found in the header.\n");
        }
        if !has_screen_window_center {
            missing.push_str("\"screenWindowCenter\" attribute not found in the header.\n");
        }
        if version.multipart || version.non_image {
            if !has_name {
                missing.push_str("\"name\" attribute not found in the header.\n");
            }
            if !has_type {
                missing.push_str("\"type\" attribute not found in the header.\n");
            }
        }

        if !missing.is_empty() {
            return Err(ExrError::new(ErrorCode::INVALID_HEADER, missing));
        }
    }

    info.header_len = (orig_size - size) as u32;

    Ok(Parsed::Header)
}

/// `EXRHeader`: what the header of a file says.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExrHeader {
    /// The `pixelAspectRatio` attribute.
    pub pixel_aspect_ratio: f32,
    /// The `lineOrder` attribute.
    pub line_order: LineOrder,
    /// The `dataWindow` attribute: the pixels the file holds.
    pub data_window: Box2i,
    /// The `displayWindow` attribute.
    pub display_window: Box2i,
    /// The `screenWindowCenter` attribute.
    pub screen_window_center: [f32; 2],
    /// The `screenWindowWidth` attribute.
    pub screen_window_width: f32,

    /// The `chunkCount` attribute of a part of a file of several parts.
    pub chunk_count: i32,

    /// Whether the image is tiled (has a `tiles` attribute).
    pub tiled: bool,
    /// The width of a tile.
    pub tile_size_x: i32,
    /// The height of a tile.
    pub tile_size_y: i32,
    /// Whether a tiled image has one level, mip levels or rip levels.
    pub tile_level_mode: TileLevelMode,
    /// How the sizes of a tiled image's further levels are rounded.
    pub tile_rounding_mode: TileRoundingMode,

    /// Whether attribute and channel names may be longer than 31 bytes.
    pub long_name: bool,
    /// Whether the image is deep.
    pub non_image: bool,
    /// Whether this is a part of a file of several parts.
    pub multipart: bool,
    /// The length of the header in bytes, as it was read.
    pub header_len: u32,

    /// The attributes the reader has no field for (up to 128 of them).
    pub custom_attributes: Vec<Attribute>,

    /// The channels, in the order the file has them (by name).
    pub channels: Vec<ChannelInfo>,

    /// The type of each channel's values as they were loaded: its type in the file until
    /// the image is read, the type that was asked for afterwards.
    pub pixel_types: Vec<PixelType>,

    /// How the pixels are compressed.
    pub compression_type: Compression,
    /// The type to read each channel's values as: its type in the file, which can be
    /// changed to `PixelType::FLOAT` for a channel of halves before the image is read.
    pub requested_pixel_types: Vec<PixelType>,

    /// The name attribute, which a multipart file's parts need.
    pub name: String,
}

/// `EXRTile`: a tile of a tiled image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExrTile {
    /// Which tile across its level it is.
    pub offset_x: i32,
    /// Which tile down its level it is.
    pub offset_y: i32,
    /// The level across it belongs to.
    pub level_x: i32,
    /// The level down it belongs to.
    pub level_y: i32,

    /// The actual width in a tile.
    pub width: i32,
    /// The actual height in a tile.
    pub height: i32,

    /// The tile's values, channel by channel, each `tile_size_x` by `tile_size_y` of the
    /// header (of which `width` by `height` are the tile's).
    pub images: Vec<ChannelData>,
}

/// `EXRImage`: the pixels of a file: channel by channel for a file of scanlines, tile by
/// tile for a tiled one, whose further levels follow in `next_level`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExrImage {
    /// Tiled pixel data: the tiles of this level.
    pub tiles: Vec<ExrTile>,
    /// The next level of a tiled image with mip or rip levels.
    pub next_level: Option<Box<ExrImage>>,
    /// The level across this is, for a tiled image.
    pub level_x: i32,
    /// The level down this is, for a tiled image.
    pub level_y: i32,

    /// The values of each channel, for a file of scanlines: `width` by `height`.
    pub images: Vec<ChannelData>,

    /// The width of the image (or of this level) in pixels.
    pub width: i32,
    /// The height of the image (or of this level) in pixels.
    pub height: i32,
    /// How many channels it has.
    pub num_channels: i32,
}

impl ExrImage {
    /// Returns an iterator over this image and every subsequent mip or rip level chained via
    /// [`next_level`](Self::next_level).
    #[must_use]
    pub fn levels(&self) -> LevelIter<'_> {
        LevelIter { next: Some(self) }
    }

    /// Detaches and drops all chained [`next_level`](Self::next_level) images iteratively in
    /// `O(1)` stack space.
    ///
    /// (`ExrImage` keeps default drop glue rather than implementing [`Drop`] directly so that
    /// struct update syntax `ExrImage { ..ExrImage::default() }` remains valid on public
    /// non-`Copy` fields.)
    pub fn clear_levels(&mut self) {
        let mut cur = self.next_level.take();
        while let Some(mut node) = cur {
            cur = node.next_level.take();
        }
    }
}

/// Iterator over the levels of an [`ExrImage`], created by [`ExrImage::levels`].
#[derive(Clone, Debug)]
pub struct LevelIter<'a> {
    next: Option<&'a ExrImage>,
}

impl<'a> Iterator for LevelIter<'a> {
    type Item = &'a ExrImage;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next?;
        self.next = cur.next_level.as_deref();
        Some(cur)
    }
}

impl core::iter::FusedIterator for LevelIter<'_> {}

/// `ParseEXRVersionFromMemory`: the eight bytes a file starts with. The error is the
/// original's code.
///
/// # Errors
///
/// A file that does not start with OpenEXR's magic number, or is not of version 2.
pub fn parse_version(memory: &[u8]) -> Result<ExrVersion, ErrorCode> {
    if memory.len() < 8 {
        return Err(ErrorCode::INVALID_DATA);
    }

    if memory[..4] != [0x76, 0x2f, 0x31, 0x01] {
        return Err(ErrorCode::INVALID_MAGIC_NUMBER);
    }

    if memory[4] != 2 {
        return Err(ErrorCode::INVALID_EXR_VERSION);
    }

    let flags = memory[5];
    Ok(ExrVersion {
        version: 2,
        tiled: flags & 0x2 != 0,
        long_name: flags & 0x4 != 0,
        non_image: flags & 0x8 != 0,
        multipart: flags & 0x10 != 0,
    })
}

/// Whether `memory` starts with the version of a `.exr` file (`IsEXRFromMemory`).
#[must_use]
pub fn is_exr(memory: &[u8]) -> bool {
    parse_version(memory).is_ok()
}

/// `ConvertHeader`: a parsed header as an `ExrHeader`, or why not, where its type attribute
/// and its tiles disagree.
pub(crate) fn convert_header(info: &HeaderInfo) -> Result<ExrHeader, &'static str> {
    let mut exr_header = ExrHeader {
        pixel_aspect_ratio: info.pixel_aspect_ratio,
        screen_window_center: info.screen_window_center,
        screen_window_width: info.screen_window_width,
        chunk_count: info.chunk_count,
        display_window: info.display_window,
        data_window: info.data_window,
        line_order: info.line_order,
        compression_type: info.compression_type,
        tiled: info.tiled,
        tile_size_x: info.tile_size_x,
        tile_size_y: info.tile_size_y,
        tile_level_mode: info.tile_level_mode,
        tile_rounding_mode: info.tile_rounding_mode,
        name: info.name.clone(),
        ..ExrHeader::default()
    };

    match info.type_name.as_str() {
        "scanlineimage" => {
            if exr_header.tiled {
                return Err("(ConvertHeader) tiled bit must be off for `scanlineimage` type.\n");
            }
        }
        "tiledimage" => {
            if !exr_header.tiled {
                return Err("(ConvertHeader) tiled bit must be on for `tiledimage` type.\n");
            }
        }
        "deeptile" => {
            exr_header.non_image = true;
            if !exr_header.tiled {
                return Err("(ConvertHeader) tiled bit must be on for `deeptile` type.\n");
            }
        }
        "deepscanline" => exr_header.non_image = true,
        _ => {}
    }

    exr_header.channels.clone_from(&info.channels);

    exr_header.pixel_types = info
        .channels
        .iter()
        .map(|channel| channel.pixel_type)
        .collect();

    exr_header
        .requested_pixel_types
        .clone_from(&exr_header.pixel_types);

    exr_header.custom_attributes = info
        .attributes
        .iter()
        .take(MAX_CUSTOM_ATTRIBUTES)
        .cloned()
        .collect();

    exr_header.header_len = info.header_len;

    Ok(exr_header)
}

/// `ParseEXRHeaderFromMemory`: the header of a file of one part.
///
/// # Errors
///
/// A header that is damaged, lacks a required attribute or uses an unknown compression, with
/// tinyexr's error code and message.
pub fn parse_header(version: ExrVersion, memory: &[u8]) -> Result<ExrHeader, ExrError> {
    parse_header_with(version, memory, false)
}

/// `ParseEXRHeaderFromMemory` exactly as tinyexr has it, which refuses DWAA and DWAB
/// ("Unknown compression type.").
///
/// # Errors
///
/// As [`parse_header`], and DWAA or DWAB compression.
pub fn parse_header_as_tinyexr(version: ExrVersion, memory: &[u8]) -> Result<ExrHeader, ExrError> {
    parse_header_with(version, memory, true)
}

fn parse_header_with(
    version: ExrVersion,
    memory: &[u8],
    as_tinyexr: bool,
) -> Result<ExrHeader, ExrError> {
    if memory.len() < 8 {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Insufficient header/data size.\n",
        ));
    }

    let mut info = HeaderInfo::default();
    let parsed = parse_exr_header(&mut info, version, &memory[8..], as_tinyexr);
    let mut exr_header = match (parsed, convert_header(&info)) {
        (Ok(_), Ok(header)) => header,
        (Err(error), Ok(_)) => return Err(error),
        (_, Err(message)) => return Err(ExrError::new(ErrorCode::INVALID_HEADER, message)),
    };
    exr_header.multipart = version.multipart;
    exr_header.non_image = version.non_image;
    Ok(exr_header)
}

/// `LoadEXRImageFromMemory`: the pixels of the file whose header `exr_header` is. Each
/// channel comes as the type `exr_header.requested_pixel_types` asks for; afterwards
/// `exr_header.pixel_types` says the same.
///
/// # Errors
///
/// A header not given by [`parse_header`], or pixels that are damaged or too large, with tinyexr's
/// error code and message.
pub fn load_image(exr_header: &mut ExrHeader, memory: &[u8]) -> Result<ExrImage, ExrError> {
    load_image_with(exr_header, memory, false, 1)
}

/// Multi-threaded [`load_image`] decompressing scanline blocks or tiles across `num_threads`
/// worker threads (`0` selects [`std::thread::available_parallelism`], `1` runs on the
/// calling thread).
///
/// Produces a **bit-identical** [`ExrImage`] to [`load_image`] across any thread count.
///
/// # Errors
///
/// As [`load_image`].
#[cfg(feature = "std")]
pub fn load_image_parallel(
    exr_header: &mut ExrHeader,
    memory: &[u8],
    num_threads: usize,
) -> Result<ExrImage, ExrError> {
    load_image_with(exr_header, memory, false, num_threads)
}

/// `LoadEXRImageFromMemory` exactly as tinyexr has it, for every file.
///
/// # Errors
///
/// As [`load_image`].
pub fn load_image_as_tinyexr(
    exr_header: &mut ExrHeader,
    memory: &[u8],
) -> Result<ExrImage, ExrError> {
    load_image_with(exr_header, memory, true, 1)
}

fn load_image_with(
    exr_header: &mut ExrHeader,
    memory: &[u8],
    as_tinyexr: bool,
    num_threads: usize,
) -> Result<ExrImage, ExrError> {
    if memory.len() < 8 {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: "Invalid argument for LoadEXRImageFromMemory".to_string(),
        });
    }

    if exr_header.header_len == 0 {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: "EXRHeader variable is not initialized.".to_string(),
        });
    }

    let Some(marker) = (exr_header.header_len as usize).checked_add(8) else {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Insufficient data size in offset table.",
        ));
    };
    chunk::decode_exr_image_with_threads(exr_header, memory, marker, as_tinyexr, num_threads)
}

/// A value of a channel as a float, as the original reads every channel's memory.
#[inline]
fn value_as_float(image: &ChannelData, index: usize) -> f32 {
    match image {
        ChannelData::Float(values) => values[index],
        ChannelData::Uint(values) => f32::from_bits(values[index]),
        ChannelData::Half(values) => half_to_float(values[index]),
    }
}

/// An image as floats of red, green, blue and alpha, as [`load_rgba`] gives it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RgbaImage {
    /// The width in pixels.
    pub width: i32,
    /// The height in pixels.
    pub height: i32,
    /// Four floats a pixel (red, green, blue, alpha), line by line from the top.
    pub data: Vec<f32>,
}

/// `LoadEXRFromMemory`: a file as floats of red, green, blue and alpha. A file of one
/// channel gives that channel four times; any other needs channels named `R`, `G` and `B`,
/// and its alpha is 1 where it has no `A`.
///
/// # Errors
///
/// A file [`load_image`] refuses, or one with several channels but no `R`, `G` and `B`, with
/// tinyexr's error code and message.
pub fn load_rgba(memory: &[u8]) -> Result<RgbaImage, ExrError> {
    load_rgba_with(memory, false, 1)
}

/// Multi-threaded [`load_rgba`] decompressing chunks across `num_threads` worker threads
/// (`0` selects [`std::thread::available_parallelism`], `1` runs on the calling thread).
///
/// # Errors
///
/// As [`load_rgba`].
#[cfg(feature = "std")]
pub fn load_rgba_parallel(memory: &[u8], num_threads: usize) -> Result<RgbaImage, ExrError> {
    load_rgba_with(memory, false, num_threads)
}

/// `LoadEXRFromMemory` exactly as tinyexr has it
/// ([`load_image_as_tinyexr`] says where that differs).
///
/// # Errors
///
/// As [`load_rgba`].
pub fn load_rgba_as_tinyexr(memory: &[u8]) -> Result<RgbaImage, ExrError> {
    load_rgba_with(memory, true, 1)
}

#[allow(clippy::too_many_lines)]
fn load_rgba_with(
    memory: &[u8],
    as_tinyexr: bool,
    num_threads: usize,
) -> Result<RgbaImage, ExrError> {
    let exr_version = match parse_version(memory) {
        Ok(version) => version,
        Err(code) => {
            return Err(ExrError {
                code,
                message: format!("Failed to parse EXR version. code({})", code.0),
            });
        }
    };

    let mut exr_header = parse_header_with(exr_version, memory, as_tinyexr)?;

    for i in 0..exr_header.channels.len() {
        if exr_header.pixel_types[i] == PixelType::HALF {
            exr_header.requested_pixel_types[i] = PixelType::FLOAT;
        }
    }

    let exr_image = load_image_with(&mut exr_header, memory, as_tinyexr, num_threads)?;

    let mut idx_r = -1i32;
    let mut idx_g = -1i32;
    let mut idx_b = -1i32;
    let mut idx_a = -1i32;
    for (c, channel) in exr_header.channels.iter().enumerate() {
        match channel.name.as_str() {
            "R" => idx_r = c as i32,
            "G" => idx_g = c as i32,
            "B" => idx_b = c as i32,
            "A" => idx_a = c as i32,
            _ => {}
        }
    }

    let single = exr_header.channels.len() == 1;
    if !single {
        if idx_r == -1 {
            return Err(ExrError {
                code: ErrorCode::INVALID_DATA,
                message: "R channel not found".to_string(),
            });
        }
        if idx_g == -1 {
            return Err(ExrError {
                code: ErrorCode::INVALID_DATA,
                message: "G channel not found".to_string(),
            });
        }
        if idx_b == -1 {
            return Err(ExrError {
                code: ErrorCode::INVALID_DATA,
                message: "B channel not found".to_string(),
            });
        }
    }
    let sources: [Option<usize>; 4] = if single {
        [Some(0); 4]
    } else {
        [
            Some(idx_r as usize),
            Some(idx_g as usize),
            Some(idx_b as usize),
            if idx_a == -1 {
                None
            } else {
                Some(idx_a as usize)
            },
        ]
    };

    let (image_width, image_height) = (
        exr_image.width.max(0) as usize,
        exr_image.height.max(0) as usize,
    );
    let Some(pixels) = image_width.checked_mul(image_height) else {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "RGBA image dimensions overflow.\n",
        ));
    };
    let Some(rgba_len) = pixels.checked_mul(4) else {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "RGBA image dimensions overflow.\n",
        ));
    };
    let Some(mut out_rgba) = decode::zeros::<f32>(rgba_len) else {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Failed to allocate memory for RGBA output.\n",
        ));
    };

    if exr_header.tiled {
        let tile_size_x = exr_header.tile_size_x as usize;
        let tile_size_y = exr_header.tile_size_y as usize;
        for tile in &exr_image.tiles {
            for j in 0..tile_size_y {
                for i in 0..tile_size_x {
                    let ii = (tile.offset_x as isize as usize)
                        .wrapping_mul(tile_size_x)
                        .wrapping_add(i);
                    let jj = (tile.offset_y as isize as usize)
                        .wrapping_mul(tile_size_y)
                        .wrapping_add(j);

                    if ii >= image_width || jj >= image_height {
                        continue;
                    }
                    let idx = ii + jj * image_width;
                    let src_idx = i + j * tile_size_x;
                    for (k, source) in sources.iter().enumerate() {
                        out_rgba[4 * idx + k] = match source {
                            Some(c) => value_as_float(&tile.images[*c], src_idx),
                            None => 1.0,
                        };
                    }
                }
            }
        }
    } else {
        for i in 0..pixels {
            for (k, source) in sources.iter().enumerate() {
                out_rgba[4 * i + k] = match source {
                    Some(c) => value_as_float(&exr_image.images[*c], i),
                    None => 1.0,
                };
            }
        }
    }

    Ok(RgbaImage {
        width: exr_image.width,
        height: exr_image.height,
        data: out_rgba,
    })
}

/// `LoadEXR`: reads a `.exr` file from `path` as floats of red, green, blue and alpha.
///
/// # Errors
///
/// [`ErrorCode::INVALID_FILE`] if `path` cannot be read, or any [`ExrError`] returned by
/// [`load_rgba`].
#[cfg(feature = "std")]
pub fn load_rgba_file(path: impl AsRef<std::path::Path>) -> Result<RgbaImage, ExrError> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|err| ExrError::new(ErrorCode::INVALID_FILE, err.to_string()))?;
    load_rgba(&bytes)
}

/// `SaveEXR`: encodes `data` (`1`, `3`, or `4` float components per pixel) and writes the
/// `.exr` file to `path`.
///
/// # Errors
///
/// Any [`ExrError`] returned by [`save_floats`], or [`ErrorCode::SERIALIZATION_FAILED`] if
/// writing `path` fails.
#[cfg(feature = "std")]
pub fn save_floats_file(
    path: impl AsRef<std::path::Path>,
    data: &[f32],
    width: i32,
    height: i32,
    components: i32,
    save_as_fp16: bool,
) -> Result<(), ExrError> {
    let bytes = save_floats(data, width, height, components, save_as_fp16)?;
    std::fs::write(path.as_ref(), &bytes)
        .map_err(|err| ExrError::new(ErrorCode::SERIALIZATION_FAILED, err.to_string()))
}

/// Reads a single-part `.exr` image and its header from any [`std::io::Read`] stream.
///
/// # Errors
///
/// [`ErrorCode::INVALID_FILE`] on an I/O read failure, or any [`ExrError`] returned by
/// [`parse_version`], [`parse_header`], or [`load_image`].
#[cfg(feature = "std")]
pub fn read_from(mut reader: impl std::io::Read) -> Result<(ExrHeader, ExrImage), ExrError> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|err| ExrError::new(ErrorCode::INVALID_FILE, err.to_string()))?;
    let version = parse_version(&buf).map_err(|code| {
        ExrError::new(
            code,
            format!("Failed to parse EXR version. code({})", code.0),
        )
    })?;
    let mut header = parse_header(version, &buf)?;
    let image = load_image(&mut header, &buf)?;
    Ok((header, image))
}

/// Encodes `exr_image` with `exr_header` via [`save_image`] and writes the `.exr` bytes to
/// `writer`.
///
/// # Errors
///
/// Any [`ExrError`] returned by [`save_image`], or [`ErrorCode::SERIALIZATION_FAILED`] on an
/// I/O write error.
#[cfg(feature = "std")]
pub fn write_to(
    exr_image: &ExrImage,
    exr_header: &ExrHeader,
    mut writer: impl std::io::Write,
) -> Result<(), ExrError> {
    let bytes = save_image(exr_image, exr_header)?;
    writer
        .write_all(&bytes)
        .map_err(|err| ExrError::new(ErrorCode::SERIALIZATION_FAILED, err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: u32 = 0x3f80_0000;

    fn attribute(file: &mut Vec<u8>, name: &str, type_name: &str, value: &[u8]) {
        file.extend_from_slice(name.as_bytes());
        file.push(0);
        file.extend_from_slice(type_name.as_bytes());
        file.push(0);
        file.extend_from_slice(&(value.len() as u32).to_le_bytes());
        file.extend_from_slice(value);
    }

    fn file_by_hand(pixels: &[[u16; 3]; 4]) -> Vec<u8> {
        let mut file = vec![0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0];
        let mut channels = Vec::new();
        for name in ["B", "G", "R"] {
            channels.extend_from_slice(name.as_bytes());
            channels.push(0);
            channels.extend_from_slice(&PixelType::HALF.0.to_le_bytes());
            channels.extend_from_slice(&[0, 0, 0, 0]);
            channels.extend_from_slice(&1i32.to_le_bytes());
            channels.extend_from_slice(&1i32.to_le_bytes());
        }
        channels.push(0);
        let window: Vec<u8> = [0i32, 0, 1, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        attribute(&mut file, "channels", "chlist", &channels);
        attribute(&mut file, "compression", "compression", &[0]);
        attribute(&mut file, "dataWindow", "box2i", &window);
        attribute(&mut file, "displayWindow", "box2i", &window);
        attribute(&mut file, "lineOrder", "lineOrder", &[0]);
        attribute(&mut file, "pixelAspectRatio", "float", &1f32.to_le_bytes());
        attribute(&mut file, "screenWindowCenter", "v2f", &[0; 8]);
        attribute(&mut file, "screenWindowWidth", "float", &1f32.to_le_bytes());
        file.push(0);
        let first = file.len() + 16;
        let block = 8 + 3 * 2 * 2;
        file.extend_from_slice(&(first as u64).to_le_bytes());
        file.extend_from_slice(&((first + block) as u64).to_le_bytes());
        for y in 0..2usize {
            file.extend_from_slice(&(y as i32).to_le_bytes());
            file.extend_from_slice(&12i32.to_le_bytes());
            for channel in [2, 1, 0] {
                for x in 0..2 {
                    file.extend_from_slice(&pixels[y * 2 + x][channel].to_le_bytes());
                }
            }
        }
        file
    }

    #[test]
    fn a_file_put_together_by_hand_is_read() {
        let (one, half, two) = (0x3c00, 0x3800, 0x4000);
        let pixels = [[one, half, two], [0, 0, 0], [two, two, two], [half, one, 0]];
        let file = file_by_hand(&pixels);

        let RgbaImage {
            width,
            height,
            data: rgba,
        } = load_rgba(&file).unwrap();
        assert_eq!((width, height), (2, 2));
        assert_eq!(
            rgba,
            [
                1.0, 0.5, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.5, 1.0, 0.0, 1.0
            ]
        );

        let version = parse_version(&file).unwrap();
        assert!(!version.tiled && !version.multipart && !version.non_image);
        let mut header = parse_header(version, &file).unwrap();
        assert_eq!(
            header
                .channels
                .iter()
                .map(|channel| channel.name.as_str())
                .collect::<Vec<_>>(),
            ["B", "G", "R"]
        );
        assert_eq!(header.compression_type, Compression::NONE);
        let image = load_image(&mut header, &file).unwrap();
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.images[2], ChannelData::Half(vec![one, 0, two, half]));
        assert_eq!(image.images[0], ChannelData::Half(vec![two, 0, two, 0]));
    }

    #[test]
    fn a_file_that_is_not_one_is_refused_with_tinyexrs_code() {
        assert_eq!(
            parse_version(b"\x89PNG\r\n\x1a\n"),
            Err(ErrorCode::INVALID_MAGIC_NUMBER)
        );
        assert_eq!(
            parse_version(&[0x76, 0x2f, 0x31]),
            Err(ErrorCode::INVALID_DATA)
        );
        let error = load_rgba(b"not an .exr file").unwrap_err();
        assert_eq!(error.code, ErrorCode::INVALID_MAGIC_NUMBER);
        assert_eq!(error.message, "Failed to parse EXR version. code(-1)");

        let file = file_by_hand(&[[0; 3]; 4]);
        let error = load_rgba(&file[..file.len() - 4]).unwrap_err();
        assert_eq!(error.code, ErrorCode::INVALID_DATA);
    }

    fn image_of(
        width: i32,
        height: i32,
        compression_type: Compression,
        values: impl Fn(usize, usize) -> ChannelData,
        file_type: PixelType,
        memory_type: PixelType,
    ) -> (ExrImage, ExrHeader) {
        let image = ExrImage {
            images: (0..3)
                .map(|c| values(c, (width * height) as usize))
                .collect(),
            width,
            height,
            num_channels: 3,
            ..ExrImage::default()
        };
        let header = ExrHeader {
            channels: ["B", "G", "R"]
                .iter()
                .map(|name| ChannelInfo {
                    name: name.to_string(),
                    pixel_type: file_type,
                    x_sampling: 1,
                    y_sampling: 1,
                    p_linear: 0,
                })
                .collect(),
            pixel_types: vec![memory_type; 3],
            requested_pixel_types: vec![file_type; 3],
            compression_type,
            ..ExrHeader::default()
        };
        (image, header)
    }

    fn read(file: &[u8], as_tinyexr: bool) -> ExrImage {
        let version = parse_version(file).unwrap();
        let mut header = parse_header(version, file).unwrap();
        if as_tinyexr {
            load_image_as_tinyexr(&mut header, file).unwrap()
        } else {
            load_image(&mut header, file).unwrap()
        }
    }

    #[cfg(feature = "dwa")]
    #[test]
    fn dwa_is_written_and_read_but_not_as_tinyexr() {
        let smooth = |c: usize, count: usize| {
            ChannelData::Float(
                (0..count)
                    .map(|i| {
                        ((i % 70) as f32 * 0.01 + (i / 70) as f32 * 0.02 + c as f32 * 0.1)
                            .sin()
                            .abs()
                    })
                    .collect(),
            )
        };
        for compression_type in [Compression::DWAA, Compression::DWAB] {
            let (image, mut header) = image_of(
                70,
                300,
                compression_type,
                smooth,
                PixelType::HALF,
                PixelType::FLOAT,
            );
            let file = save_image(&image, &header).unwrap();
            let RgbaImage {
                width,
                height,
                data: rgba,
            } = load_rgba(&file).unwrap();
            assert_eq!((width, height), (70, 300));
            let ChannelData::Float(red) = &image.images[2] else {
                unreachable!()
            };
            assert!(
                rgba.as_chunks::<4>()
                    .0
                    .iter()
                    .zip(red)
                    .all(|(pixel, red)| (pixel[0] - red).abs() < 0.02 && pixel[3].to_bits() == ONE),
                "compression {compression_type:?}"
            );
            let (_, plain) = image_of(
                70,
                300,
                Compression::NONE,
                smooth,
                PixelType::HALF,
                PixelType::FLOAT,
            );
            assert!(file.len() * 3 < save_image(&image, &plain).unwrap().len());

            assert!(save_image_as_tinyexr(&image, &header).is_err());
            let version = parse_version(&file).unwrap();
            let refused = parse_header_as_tinyexr(version, &file).unwrap_err();
            assert_eq!(
                (refused.code, refused.message.as_str()),
                (ErrorCode::UNSUPPORTED_FORMAT, "Unknown compression type.")
            );
            assert!(load_rgba_as_tinyexr(&file).is_err());

            header.custom_attributes = vec![dwa_compression_level_attribute(-1.0)];
            assert_eq!(
                save_image(&image, &header).unwrap_err().message,
                DWA_INVALID_LEVEL
            );
            header.custom_attributes = vec![dwa_compression_level_attribute(0.0)];
            let lossless_quantization = save_image(&image, &header).unwrap();
            assert!(lossless_quantization.len() > file.len());
        }
    }

    #[cfg(feature = "dwa")]
    #[test]
    fn dwa_honors_p_linear_in_header_and_codec() {
        let count = 32 * 32;
        let image = ExrImage {
            images: vec![ChannelData::Float(
                (0..count)
                    .map(|i| ((i % 32) as f32 * 0.03 + 0.05).min(1.0))
                    .collect(),
            )],
            width: 32,
            height: 32,
            num_channels: 1,
            ..ExrImage::default()
        };
        let make_header = |p_linear: u8| ExrHeader {
            channels: vec![ChannelInfo {
                name: "Y".to_string(),
                pixel_type: PixelType::HALF,
                x_sampling: 1,
                y_sampling: 1,
                p_linear,
            }],
            pixel_types: vec![PixelType::FLOAT],
            requested_pixel_types: vec![PixelType::HALF],
            compression_type: Compression::DWAA,
            ..ExrHeader::default()
        };

        let nonlin_bytes = save_image(&image, &make_header(0)).unwrap();
        let lin_bytes = save_image(&image, &make_header(1)).unwrap();
        assert_ne!(nonlin_bytes, lin_bytes);

        let ver = parse_version(&lin_bytes).unwrap();
        let mut parsed = parse_header(ver, &lin_bytes).unwrap();
        assert_eq!(parsed.channels[0].p_linear, 1);
        parsed.requested_pixel_types[0] = PixelType::FLOAT;
        let loaded = load_image(&mut parsed, &lin_bytes).unwrap();
        let orig = image.images[0].as_f32_slice().unwrap();
        let got = loaded.images[0].as_f32_slice().unwrap();
        assert!(orig.iter().zip(got).all(|(a, b)| (a - b).abs() < 0.05));
    }

    #[test]
    fn what_is_written_comes_back() {
        let floats = |c: usize, count: usize| {
            ChannelData::Float(
                (0..count)
                    .map(|i| ((i * 31 + c * 7) % 97) as f32 * 0.125 - 3.0)
                    .collect(),
            )
        };
        for compression_type in [
            Compression::NONE,
            Compression::RLE,
            Compression::ZIPS,
            Compression::ZIP,
            Compression::PIZ,
            Compression::B44,
            Compression::B44A,
        ] {
            let (image, header) = image_of(
                21,
                40,
                compression_type,
                floats,
                PixelType::FLOAT,
                PixelType::FLOAT,
            );
            let file = save_image(&image, &header).unwrap();
            assert_eq!(
                read(&file, false).images,
                image.images,
                "compression {compression_type:?}"
            );
            let (_, header) = image_of(
                21,
                40,
                compression_type,
                floats,
                PixelType::HALF,
                PixelType::FLOAT,
            );
            let file = save_image(&image, &header).unwrap();
            let RgbaImage {
                width,
                height,
                data: rgba,
            } = load_rgba(&file).unwrap();
            assert_eq!((width, height), (21, 40));
            if compression_type != Compression::B44 && compression_type != Compression::B44A {
                let ChannelData::Float(red) = &image.images[2] else {
                    unreachable!()
                };
                assert!(
                    rgba.as_chunks::<4>()
                        .0
                        .iter()
                        .zip(red)
                        .all(|(pixel, red)| pixel[0].to_bits() == red.to_bits()
                            && pixel[3].to_bits() == ONE),
                    "compression {compression_type:?}"
                );
            }
        }
        let halves = |c: usize, count: usize| {
            ChannelData::Half((0..count).map(|i| (i * 977 + c * 13) as u16).collect())
        };
        let (image, header) = image_of(
            21,
            40,
            Compression::PXR24,
            halves,
            PixelType::HALF,
            PixelType::HALF,
        );
        assert_eq!(
            read(&save_image(&image, &header).unwrap(), false).images,
            image.images
        );
    }

    #[test]
    fn a_block_that_does_not_come_out_smaller_is_stored_as_it_is() {
        let floats = |c: usize, count: usize| {
            ChannelData::Float((0..count).map(|i| (i * 3 + c) as f32).collect())
        };
        let (image, header) = image_of(
            5,
            3,
            Compression::B44,
            floats,
            PixelType::FLOAT,
            PixelType::FLOAT,
        );
        let file = save_image(&image, &header).unwrap();
        let as_tinyexr = save_image_as_tinyexr(&image, &header).unwrap();
        assert_eq!(file.len(), as_tinyexr.len());
        assert_ne!(file, as_tinyexr);
        let (_, plain_header) = image_of(
            5,
            3,
            Compression::NONE,
            floats,
            PixelType::FLOAT,
            PixelType::FLOAT,
        );
        let plain = save_image(&image, &plain_header).unwrap();
        assert_eq!(
            file[file.len() - 3 * 60..file.len() - 2 * 60],
            plain[plain.len() - 3 * 68 + 8..plain.len() - 2 * 68]
        );
        assert_eq!(read(&file, false).images, image.images);
        assert_eq!(read(&as_tinyexr, true).images, image.images);
        assert_ne!(read(&as_tinyexr, false).images, image.images);
        assert_ne!(read(&file, true).images, image.images);

        let halves = |c: usize, count: usize| {
            ChannelData::Half((0..count).map(|i| (i * 977 + c * 13 + 5) as u16).collect())
        };
        let (image, header) = image_of(
            2,
            1,
            Compression::PXR24,
            halves,
            PixelType::HALF,
            PixelType::HALF,
        );
        let file = save_image(&image, &header).unwrap();
        let as_tinyexr = save_image_as_tinyexr(&image, &header).unwrap();
        assert!(file.len() < as_tinyexr.len());
        assert_eq!(read(&file, false).images, image.images);
        assert_eq!(read(&as_tinyexr, true).images, image.images);
        assert_eq!(read(&as_tinyexr, false).images, image.images);
    }

    #[test]
    fn what_the_original_would_read_past_is_refused() {
        let floats = |_: usize, count: usize| ChannelData::Float(vec![0.5; count]);
        let (image, header) = image_of(
            4,
            4,
            Compression::ZIP,
            floats,
            PixelType::FLOAT,
            PixelType::FLOAT,
        );
        assert!(save_image(&image, &header).is_ok());
        let mut short = image.clone();
        short.images[1] = ChannelData::Float(vec![0.5; 15]);
        assert_eq!(
            save_image(&short, &header).unwrap_err().code,
            ErrorCode::INVALID_ARGUMENT
        );
        let mut other = header.clone();
        other.pixel_types[0] = PixelType::HALF;
        assert_eq!(
            save_image(&image, &other).unwrap_err().code,
            ErrorCode::INVALID_ARGUMENT
        );
        let mut flat = image.clone();
        flat.height = 0;
        assert_eq!(
            save_image(&flat, &header).unwrap_err().code,
            ErrorCode::INVALID_ARGUMENT
        );
        #[cfg(feature = "zfp")]
        {
            let (odd, mut zfp) = image_of(
                6,
                4,
                Compression::ZFP,
                floats,
                PixelType::FLOAT,
                PixelType::FLOAT,
            );
            assert_eq!(
                save_image(&odd, &zfp).unwrap_err().message,
                ZFP_NOT_MULTIPLE_OF_FOUR
            );
            zfp.requested_pixel_types[1] = PixelType::HALF;
            let error = save_image(&odd, &zfp).unwrap_err();
            assert_eq!(
                (error.code, error.message.as_str()),
                (
                    ErrorCode::SERIALIZATION_FAILED,
                    "Pixel type must be FLOAT for ZFP compression"
                )
            );
        }
        let error = save_floats(&[0.0; 8], 2, 2, 2, false).unwrap_err();
        assert_eq!(
            (error.code, error.message.as_str()),
            (
                ErrorCode::INVALID_ARGUMENT,
                "Unsupported component value : 2\n"
            )
        );
    }

    #[test]
    fn mismatched_requested_pixel_types_returns_invalid_argument() {
        let file = file_by_hand(&[[0x3c00; 3]; 4]);
        let version = parse_version(&file).unwrap();
        let mut header = parse_header(version, &file).unwrap();
        header.requested_pixel_types.pop();
        assert_eq!(
            load_image(&mut header, &file).unwrap_err().code,
            ErrorCode::INVALID_ARGUMENT
        );
    }

    #[test]
    fn raw_name_truncation_preserves_bytes_up_to_255_before_utf8_lossy() {
        // 253 'a's + invalid UTF-8 byte 0xFF + 'z' = 255 raw bytes.
        // Previously `from_utf8_lossy` expanded 0xFF to 3 bytes (total 257) and slicing `[..255]`
        // dropped 'z' and split the replacement codepoint!
        let mut raw = vec![b'a'; 253];
        raw.push(0xff);
        raw.push(b'z');
        raw.extend_from_slice(b"extra_beyond_255");
        let converted = bytes_to_name_string(&raw);
        assert!(converted.ends_with("\u{fffd}z"));
        assert_eq!(converted.chars().count(), 255);
    }

    #[test]
    fn error_code_display_channel_data_helpers_and_deep_level_drop() {
        for code in [
            ErrorCode::SUCCESS,
            ErrorCode::INVALID_MAGIC_NUMBER,
            ErrorCode::INVALID_EXR_VERSION,
            ErrorCode::INVALID_ARGUMENT,
            ErrorCode::INVALID_DATA,
            ErrorCode::INVALID_FILE,
            ErrorCode::INVALID_PARAMETER,
            ErrorCode::UNSUPPORTED_FORMAT,
            ErrorCode::INVALID_HEADER,
            ErrorCode::UNSUPPORTED_FEATURE,
            ErrorCode::SERIALIZATION_FAILED,
            ErrorCode::LAYER_NOT_FOUND,
            ErrorCode::DATA_TOO_LARGE,
            ErrorCode(-99),
        ] {
            assert_ne!(code.to_string(), "");
        }
        let err_trait: &dyn core::error::Error = &ErrorCode::INVALID_DATA;
        assert!(err_trait.source().is_none());

        let ch_u = ChannelData::Uint(vec![10, 20]);
        let ch_h = ChannelData::Half(vec![0x3c00, 0x3800]);
        let ch_f = ChannelData::Float(vec![1.5]);
        assert_eq!(
            (ch_u.len(), ch_u.is_empty(), ch_u.pixel_type()),
            (2, false, PixelType::UINT)
        );
        assert_eq!(ch_u.as_u32_slice(), Some(&[10, 20][..]));
        assert_eq!(ch_u.as_u16_slice(), None);
        assert_eq!(ch_u.to_f32_vec(), vec![10.0, 20.0]);
        assert_eq!(ch_h.pixel_type(), PixelType::HALF);
        assert_eq!(ch_h.as_u16_slice(), Some(&[0x3c00, 0x3800][..]));
        assert_eq!(ch_h.to_f32_vec(), vec![1.0, 0.5]);
        assert_eq!(ch_f.pixel_type(), PixelType::FLOAT);
        assert_eq!(ch_f.as_f32_slice(), Some(&[1.5][..]));
        assert!(ChannelData::Float(Vec::new()).is_empty());

        // 20,000 linked `next_level` nodes iterate cleanly and drop iteratively without stack overflow.
        let mut head = ExrImage {
            level_x: 0,
            ..ExrImage::default()
        };
        let mut chain: Option<Box<ExrImage>> = None;
        for lvl in (1..20_000).rev() {
            chain = Some(Box::new(ExrImage {
                level_x: lvl,
                next_level: chain,
                ..ExrImage::default()
            }));
        }
        head.next_level = chain;
        assert_eq!(head.levels().count(), 20_000);
        head.clear_levels();
        assert_eq!(head.levels().count(), 1);
    }

    #[test]
    fn tiled_chunk_deduplication_oom_bomb_is_refused() {
        // Craft a tiny tiled EXR whose offset table duplicates one 4096x4096 RGBA float tile
        // 32 times (32 * 256 MiB = 8 GiB decoded memory from a ~1 KB file!).
        let mut file = vec![0x76, 0x2f, 0x31, 0x01, 2, 0x02, 0, 0];
        let mut channels = Vec::new();
        for name in ["A", "B", "G", "R"] {
            channels.extend_from_slice(name.as_bytes());
            channels.push(0);
            channels.extend_from_slice(&PixelType::FLOAT.0.to_le_bytes());
            channels.extend_from_slice(&[0, 0, 0, 0]);
            channels.extend_from_slice(&1i32.to_le_bytes());
            channels.extend_from_slice(&1i32.to_le_bytes());
        }
        channels.push(0);
        // 32x1 tiles of 4096x4096 -> width = 131_072, height = 4096 (both <= DIMENSION_THRESHOLD)
        let window: Vec<u8> = [0i32, 0, 131_072 - 1, 4096 - 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut tiledesc = Vec::new();
        tiledesc.extend_from_slice(&4096i32.to_le_bytes());
        tiledesc.extend_from_slice(&4096i32.to_le_bytes());
        tiledesc.push(0); // ONE_LEVEL

        attribute(&mut file, "channels", "chlist", &channels);
        attribute(
            &mut file,
            "compression",
            "compression",
            &[Compression::RLE.0 as u8],
        );
        attribute(&mut file, "dataWindow", "box2i", &window);
        attribute(&mut file, "displayWindow", "box2i", &window);
        attribute(&mut file, "lineOrder", "lineOrder", &[0]);
        attribute(&mut file, "pixelAspectRatio", "float", &1f32.to_le_bytes());
        attribute(&mut file, "screenWindowCenter", "v2f", &[0; 8]);
        attribute(&mut file, "screenWindowWidth", "float", &1f32.to_le_bytes());
        attribute(&mut file, "tiles", "tiledesc", &tiledesc);
        file.push(0);

        let chunk_offset = (file.len() + 32 * 8) as u64;
        for _ in 0..32 {
            file.extend_from_slice(&chunk_offset.to_le_bytes());
        }
        // Minimal 24-byte tile chunk header (`level_x = 0, level_y = 0, data_len = 4`).
        file.extend_from_slice(&0i32.to_le_bytes());
        file.extend_from_slice(&0i32.to_le_bytes());
        file.extend_from_slice(&0i32.to_le_bytes());
        file.extend_from_slice(&0i32.to_le_bytes());
        file.extend_from_slice(&4i32.to_le_bytes());
        file.extend_from_slice(&[0, 0, 0, 0]);

        let ver = parse_version(&file).unwrap();
        let mut hdr = parse_header(ver, &file).unwrap();
        let err = load_image(&mut hdr, &file).unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_DATA);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_scanline_encode_and_decode_are_bit_identical() {
        let floats = |c: usize, count: usize| {
            ChannelData::Float(
                (0..count)
                    .map(|i| ((i * 31 + c * 7) % 97) as f32 * 0.125 - 1.5)
                    .collect(),
            )
        };
        let compressions = [
            Compression::NONE,
            Compression::RLE,
            Compression::ZIPS,
            Compression::ZIP,
            Compression::PIZ,
            Compression::PXR24,
            Compression::B44,
            Compression::B44A,
            #[cfg(feature = "dwa")]
            Compression::DWAA,
            #[cfg(feature = "zfp")]
            Compression::ZFP,
        ];

        for comp in compressions {
            let file_type = if comp == Compression::ZFP {
                PixelType::FLOAT
            } else {
                PixelType::HALF
            };
            let (image, mut header) = image_of(64, 96, comp, floats, file_type, PixelType::FLOAT);
            if comp == Compression::ZFP {
                header.custom_attributes = ZfpSettings::default().attributes();
            }

            let seq_bytes = save_image(&image, &header).unwrap();
            for threads in [0, 1, 2, 4] {
                let par_bytes = save_image_parallel(&image, &header, threads).unwrap();
                assert_eq!(
                    par_bytes, seq_bytes,
                    "save_image_parallel {comp:?} t={threads}"
                );

                let ver = parse_version(&seq_bytes).unwrap();
                let mut h_seq = parse_header(ver, &seq_bytes).unwrap();
                let mut h_par = h_seq.clone();
                let img_seq = load_image(&mut h_seq, &seq_bytes).unwrap();
                let img_par = load_image_parallel(&mut h_par, &seq_bytes, threads).unwrap();
                assert_eq!(img_par, img_seq, "load_image_parallel {comp:?} t={threads}");
                assert_eq!(
                    load_rgba_parallel(&seq_bytes, threads).unwrap(),
                    load_rgba(&seq_bytes).unwrap()
                );
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_tiled_encode_and_decode_are_bit_identical() {
        let mut tiled_hdr = ExrHeader {
            data_window: Box2i {
                min_x: 0,
                min_y: 0,
                max_x: 63,
                max_y: 63,
            },
            tiled: true,
            tile_size_x: 16,
            tile_size_y: 16,
            tile_level_mode: TileLevelMode::MIPMAP_LEVELS,
            channels: vec![ChannelInfo {
                name: "R".to_string(),
                pixel_type: PixelType::HALF,
                x_sampling: 1,
                y_sampling: 1,
                p_linear: 0,
            }],
            pixel_types: vec![PixelType::HALF],
            requested_pixel_types: vec![PixelType::HALF],
            compression_type: Compression::PIZ,
            ..ExrHeader::default()
        };
        let lvls = tile_levels(&tiled_hdr).unwrap();
        let mut level_imgs: Vec<ExrImage> = lvls
            .iter()
            .map(|l| ExrImage {
                level_x: l.level_x,
                level_y: l.level_y,
                width: l.width,
                height: l.height,
                num_channels: 1,
                tiles: (0..l.num_x_tiles * l.num_y_tiles)
                    .map(|t| {
                        let ox = t % l.num_x_tiles;
                        let oy = t / l.num_x_tiles;
                        ExrTile {
                            offset_x: ox,
                            offset_y: oy,
                            level_x: l.level_x,
                            level_y: l.level_y,
                            width: 16.min(l.width - ox * 16),
                            height: 16.min(l.height - oy * 16),
                            images: vec![ChannelData::Half(
                                (0..256).map(|k| (k * 17 + t as usize) as u16).collect(),
                            )],
                        }
                    })
                    .collect(),
                ..ExrImage::default()
            })
            .collect();
        let tiled_img = level_imgs
            .drain(..)
            .rev()
            .reduce(|next, mut cur| {
                cur.next_level = Some(Box::new(next));
                cur
            })
            .unwrap();
        let seq_tiled = save_image(&tiled_img, &tiled_hdr).unwrap();
        let par_tiled = save_image_parallel(&tiled_img, &tiled_hdr, 4).unwrap();
        assert_eq!(par_tiled, seq_tiled);
        let ver = parse_version(&seq_tiled).unwrap();
        tiled_hdr = parse_header(ver, &seq_tiled).unwrap();
        let mut tiled_hdr_par = tiled_hdr.clone();
        assert_eq!(
            load_image_parallel(&mut tiled_hdr_par, &seq_tiled, 4).unwrap(),
            load_image(&mut tiled_hdr, &seq_tiled).unwrap()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_and_stream_helpers_round_trip() {
        let floats = |c: usize, count: usize| {
            ChannelData::Float(
                (0..count)
                    .map(|i| ((i * 31 + c * 7) % 97) as f32 * 0.125 - 1.5)
                    .collect(),
            )
        };
        let rgba: Vec<f32> = (0..32 * 32 * 4)
            .map(|i| (i % 64) as f32 * 0.015_625)
            .collect();
        let sf_seq = save_floats(&rgba, 32, 32, 4, false).unwrap();
        let sf_par = save_floats_parallel(&rgba, 32, 32, 4, false, 4).unwrap();
        assert_eq!(sf_par, sf_seq);

        let mut stream_buf = Vec::new();
        let (s_img, s_hdr) = image_of(
            16,
            16,
            Compression::ZIP,
            floats,
            PixelType::FLOAT,
            PixelType::FLOAT,
        );
        write_to(&s_img, &s_hdr, &mut stream_buf).unwrap();
        let (r_hdr, r_img) = read_from(std::io::Cursor::new(&stream_buf)).unwrap();
        assert_eq!(r_hdr.channels.len(), 3);
        assert_eq!(r_img.images, s_img.images);

        let tmp_path = std::env::temp_dir().join("tinyexr_nrp_test_roundtrip.exr");
        save_floats_file(&tmp_path, &rgba, 32, 32, 4, false).unwrap();
        let loaded_file = load_rgba_file(&tmp_path).unwrap();
        let _ = std::fs::remove_file(&tmp_path);
        assert_eq!((loaded_file.width, loaded_file.height), (32, 32));
        assert_eq!(loaded_file.data, rgba);
    }

    #[test]
    fn header_len_extremes_and_zero_offset_reconstruction_are_safe() {
        let mut file = file_by_hand(&[[0x3c00; 3]; 4]);
        let ver = parse_version(&file).unwrap();
        let mut hdr = parse_header(ver, &file).unwrap();
        hdr.header_len = u32::MAX;
        assert_eq!(
            load_image(&mut hdr, &file).unwrap_err().code,
            ErrorCode::INVALID_DATA
        );

        let mut mp_hdrs = vec![hdr];
        assert!(load_multipart_images(&mut mp_hdrs, &file).is_err());

        // Zero out the scanline offset table to exercise `reconstruct_line_offsets`, then
        // corrupt `data_len` to `u32::MAX` to verify checked overflow rejection.
        let hdr_end = 8 + parse_header(ver, &file).unwrap().header_len as usize;
        file[hdr_end..hdr_end + 16].fill(0);
        assert!(load_rgba(&file).is_ok());
        file[hdr_end + 16 + 4..hdr_end + 16 + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(load_rgba(&file).is_err());
    }

    #[test]
    fn the_levels_of_a_tiled_image() {
        let mut header = ExrHeader {
            data_window: Box2i {
                min_x: 0,
                min_y: 0,
                max_x: 9,
                max_y: 4,
            },
            tile_size_x: 4,
            tile_size_y: 4,
            tile_level_mode: TileLevelMode::ONE_LEVEL,
            ..ExrHeader::default()
        };
        assert_eq!(
            tile_levels(&header).unwrap(),
            [TileLevel {
                level_x: 0,
                level_y: 0,
                width: 10,
                height: 5,
                num_x_tiles: 3,
                num_y_tiles: 2
            }]
        );
        header.tile_level_mode = TileLevelMode::MIPMAP_LEVELS;
        let sizes = |header: &ExrHeader| {
            tile_levels(header)
                .unwrap()
                .iter()
                .map(|level| (level.width, level.height))
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(&header), [(10, 5), (5, 2), (2, 1), (1, 1)]);
        header.tile_rounding_mode = TileRoundingMode::ROUND_UP;
        assert_eq!(sizes(&header), [(10, 5), (5, 3), (3, 2), (2, 1), (1, 1)]);
        header.tile_level_mode = TileLevelMode::RIPMAP_LEVELS;
        header.tile_rounding_mode = TileRoundingMode::ROUND_DOWN;
        let levels = tile_levels(&header).unwrap();
        assert_eq!(levels.len(), 4 * 3);
        assert_eq!(
            (
                levels[1].level_x,
                levels[1].level_y,
                levels[1].width,
                levels[1].height
            ),
            (1, 0, 5, 5)
        );
        assert_eq!(
            (
                levels[11].level_x,
                levels[11].level_y,
                levels[11].width,
                levels[11].height
            ),
            (3, 2, 1, 1)
        );
        header.tile_size_x = 0;
        assert!(tile_levels(&header).is_none());
    }

    #[test]
    fn halves_and_floats() {
        for (half, float) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x0000, 0.0),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ] {
            assert_eq!(half_to_float(half).to_bits(), float.to_bits());
            assert_eq!(float_to_half(float), half);
        }
        assert!(half_to_float(0x7e00).is_nan());
        assert_eq!(float_to_half(1e10), 0x7c00);
    }
}

/// (The README's examples, compiled and run with the tests.)
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// (And those of the Guide.)
#[cfg(doctest)]
#[doc = include_str!("../docs/guide.md")]
struct GuideDoctests;
