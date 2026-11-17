//! Synthetic data, the same from a seed here as in the C side of the comparison
//! (`compare/`), and the dumps the tests run as well as the comparison (`examples/dump.rs`).

// (The arithmetic is written as the C side has it; not every user of this module uses all of
// it.)
#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::manual_checked_ops,
    dead_code
)]

/// The two programs make the same blocks from the same seed: a 64-bit linear congruential
/// generator, the high half of its state.
pub struct Random {
    pub state: u64,
}

impl Random {
    pub fn next(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }
}

/// FNV-1a, 64 bits.
pub struct Hash {
    pub value: u64,
}

impl Hash {
    pub fn new() -> Hash {
        Hash {
            value: 14695981039346656037,
        }
    }

    pub fn add(&mut self, data: &[u8]) {
        for &byte in data {
            self.value ^= u64::from(byte);
            self.value = self.value.wrapping_mul(1099511628211);
        }
    }
}

/// Whether the comparison leaves a file with this header unread. A damaged size can ask for
/// more memory and time than there is; those are left, but for the sizes tinyexr refuses at
/// once. Two more are left, which stop tinyexr: a tile size of 0, which it divides by, and a
/// count of chunks in the thousands of millions, which it makes a table for before it looks
/// whether the file has one.
pub fn tinyexr_left(header: &tinyexr_nrp::ExrHeader) -> bool {
    let limit = 1024 * 8192;
    let w = i64::from(header.data_window.max_x) - i64::from(header.data_window.min_x) + 1;
    let h = i64::from(header.data_window.max_y) - i64::from(header.data_window.min_y) + 1;
    let (tile_x, tile_y) = (i64::from(header.tile_size_x), i64::from(header.tile_size_y));
    let refused = w <= 0
        || h <= 0
        || w > limit
        || h > limit
        || (header.tiled != 0 && (tile_x > limit || tile_y > limit));
    if refused {
        false
    } else if header.tiled != 0 {
        w * h > (1 << 22)
            || tile_x <= 0
            || tile_y <= 0
            || tile_x * tile_y > (1 << 16)
            || (w / tile_x + 1) * (h / tile_y + 1) * tile_x * tile_y > (1 << 23)
    } else {
        w * h > (1 << 22) || header.chunk_count > (1 << 24)
    }
}

/// The `tinyexr_read` dumps: the files `compare/dump_tinyexr.cc` made with tinyexr and wrote
/// to a file (for each: whether halves are to be read as floats, and the file), each read as
/// that program reads it with tinyexr: its version, its header, its image, and it as floats
/// of red, green, blue and alpha.
pub fn tinyexr_read(name: &str, count: u32, file: &[u8]) -> String {
    use tinyexr_nrp::{self as tinyexr, ChannelData, ExrError, ExrImage};
    let mut hash = Hash::new();
    let (mut headers, mut images, mut refused, mut left, mut as_rgba, mut stopped) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let mut file_total = 0u64;
    let mut out = String::new();
    let mut at = 0;
    let number = |at: &mut usize| {
        let value = u32::from_le_bytes(file[*at..*at + 4].try_into().unwrap()) as usize;
        *at += 4;
        value
    };
    fn add_number(hash: &mut Hash, value: impl Into<i32>) {
        let value: i32 = value.into();
        hash.add(&value.to_le_bytes());
    }
    // A name with its end, or a mark in its place where it was not UTF-8.
    fn add_name(hash: &mut Hash, text: &str) {
        if text.contains('\u{fffd}') {
            hash.add(&0xfffffffdu32.to_le_bytes());
        } else {
            hash.add(text.as_bytes());
            hash.add(&[0]);
        }
    }
    fn add_message(hash: &mut Hash, message: &str) {
        if message.is_empty() {
            hash.add(&0xfffffffeu32.to_le_bytes());
        } else {
            hash.add(message.as_bytes());
            hash.add(&[0]);
        }
    }
    fn add_error(hash: &mut Hash, error: &ExrError) {
        add_number(hash, error.code);
        add_message(hash, &error.message);
    }
    fn add_values(hash: &mut Hash, values: &ChannelData) {
        match values {
            ChannelData::Uint(values) => values
                .iter()
                .for_each(|value| hash.add(&value.to_le_bytes())),
            ChannelData::Half(values) => values
                .iter()
                .for_each(|value| hash.add(&value.to_le_bytes())),
            ChannelData::Float(values) => values
                .iter()
                .for_each(|value| hash.add(&value.to_bits().to_le_bytes())),
        }
    }
    // The file as floats of red, green, blue and alpha.
    fn continue_with_rgba(hash: &mut Hash, wish: usize, bytes: &[u8], as_rgba: &mut u32) {
        let loaded = tinyexr::load_rgba_as_tinyexr(bytes);
        if wish & 8 != 0 {
            hash.add(&0xfffffffcu32.to_le_bytes());
            if loaded.is_ok() {
                hash.add(b"read where tinyexr stops");
            }
            return;
        }
        match loaded {
            Err(error) => add_error(hash, &error),
            Ok(tinyexr::RgbaImage {
                width,
                height,
                data: rgba,
            }) => {
                add_number(hash, 0);
                add_message(hash, "");
                *as_rgba += 1;
                add_number(hash, width);
                add_number(hash, height);
                for value in &rgba {
                    hash.add(&value.to_bits().to_le_bytes());
                }
            }
        }
    }
    for i in 0..count {
        let wish = number(&mut at);
        let size = number(&mut at);
        let bytes = &file[at..at + size];
        at += size;
        file_total += size as u64;

        add_number(&mut hash, size as i32);
        hash.add(bytes);

        let mut large = false;
        match tinyexr::parse_version(bytes) {
            Err(code) => add_number(&mut hash, code),
            Ok(version) => {
                add_number(&mut hash, 0);
                add_number(&mut hash, version.version);
                add_number(&mut hash, i32::from(version.tiled));
                add_number(&mut hash, i32::from(version.long_name));
                add_number(&mut hash, i32::from(version.non_image));
                add_number(&mut hash, i32::from(version.multipart));

                match tinyexr::parse_header_as_tinyexr(&version, bytes) {
                    Err(error) => add_error(&mut hash, &error),
                    Ok(mut header) => {
                        add_number(&mut hash, 0);
                        add_message(&mut hash, "");
                        headers += 1;
                        hash.add(&header.pixel_aspect_ratio.to_bits().to_le_bytes());
                        add_number(&mut hash, header.line_order);
                        for window in [header.data_window, header.display_window] {
                            add_number(&mut hash, window.min_x);
                            add_number(&mut hash, window.min_y);
                            add_number(&mut hash, window.max_x);
                            add_number(&mut hash, window.max_y);
                        }
                        hash.add(&header.screen_window_center[0].to_bits().to_le_bytes());
                        hash.add(&header.screen_window_center[1].to_bits().to_le_bytes());
                        hash.add(&header.screen_window_width.to_bits().to_le_bytes());
                        add_number(&mut hash, header.chunk_count);
                        add_number(&mut hash, header.tiled);
                        add_number(&mut hash, header.tile_size_x);
                        add_number(&mut hash, header.tile_size_y);
                        add_number(&mut hash, header.tile_level_mode);
                        add_number(&mut hash, header.tile_rounding_mode);
                        add_number(&mut hash, header.long_name);
                        add_number(&mut hash, header.non_image);
                        add_number(&mut hash, header.multipart);
                        add_number(&mut hash, header.header_len as i32);
                        add_number(&mut hash, header.custom_attributes.len() as i32);
                        for attribute in &header.custom_attributes {
                            add_name(&mut hash, &attribute.name);
                            add_name(&mut hash, &attribute.type_name);
                            add_number(&mut hash, attribute.value.len() as i32);
                            hash.add(&attribute.value);
                        }
                        add_number(&mut hash, header.channels.len() as i32);
                        for (c, channel) in header.channels.iter().enumerate() {
                            add_name(&mut hash, &channel.name);
                            add_number(&mut hash, channel.pixel_type);
                            add_number(&mut hash, channel.x_sampling);
                            add_number(&mut hash, channel.y_sampling);
                            add_number(&mut hash, i32::from(channel.p_linear));
                            add_number(&mut hash, header.pixel_types[c]);
                            add_number(&mut hash, header.requested_pixel_types[c]);
                        }
                        add_number(&mut hash, header.compression_type);
                        add_name(&mut hash, &header.name);

                        large = tinyexr_left(&header);
                        if large {
                            left += 1;
                        } else {
                            // (A damaged file that is not compressed is read with halves as
                            // floats: tinyexr reads its halves as halves without a look at
                            // the length of a block, past the end of the file.)
                            if wish & 1 != 0
                                || (wish & 2 != 0
                                    && header.compression_type == tinyexr::Compression::NONE)
                            {
                                for c in 0..header.channels.len() {
                                    if header.pixel_types[c] == tinyexr::PixelType::HALF {
                                        header.requested_pixel_types[c] = tinyexr::PixelType::FLOAT;
                                    }
                                }
                            }
                            let loaded = tinyexr::load_image_as_tinyexr(&mut header, bytes);
                            match (wish & 4 != 0, loaded) {
                                // Where tinyexr stops (it throws, and nothing catches it),
                                // the port has to refuse the file.
                                (true, loaded) => {
                                    hash.add(&0xfffffffcu32.to_le_bytes());
                                    stopped += 1;
                                    if loaded.is_ok() {
                                        hash.add(b"read where tinyexr stops");
                                    }
                                }
                                (false, Err(error)) => {
                                    add_error(&mut hash, &error);
                                    refused += 1;
                                }
                                (false, Ok(image)) => {
                                    add_number(&mut hash, 0);
                                    add_message(&mut hash, "");
                                    images += 1;
                                    for &pixel_type in &header.pixel_types {
                                        add_number(&mut hash, pixel_type);
                                    }
                                    let mut next: Option<&ExrImage> = Some(&image);
                                    while let Some(level) = next {
                                        add_number(&mut hash, level.level_x);
                                        add_number(&mut hash, level.level_y);
                                        add_number(&mut hash, level.width);
                                        add_number(&mut hash, level.height);
                                        add_number(&mut hash, level.num_channels);
                                        add_number(&mut hash, level.tiles.len() as i32);
                                        if !level.tiles.is_empty() {
                                            for tile in &level.tiles {
                                                add_number(&mut hash, tile.offset_x);
                                                add_number(&mut hash, tile.offset_y);
                                                add_number(&mut hash, tile.level_x);
                                                add_number(&mut hash, tile.level_y);
                                                add_number(&mut hash, tile.width);
                                                add_number(&mut hash, tile.height);
                                                for values in &tile.images {
                                                    add_values(&mut hash, values);
                                                }
                                            }
                                        } else {
                                            for values in &level.images {
                                                add_values(&mut hash, values);
                                            }
                                        }
                                        next = level.next_level.as_deref();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !large {
            continue_with_rgba(&mut hash, wish, bytes, &mut as_rgba);
        }

        if (i + 1) % 200 == 0 || i + 1 == count {
            out += &format!("{name} {} {:016x}\n", i + 1, hash.value);
        }
    }
    out + &format!(
        "{name} bytes of files: {file_total}; headers read: {headers}, images read: {images}, refused: {refused}, left: {left}, where tinyexr stops: {stopped}; as red, green, blue and alpha: {as_rgba}\n"
    )
}

/// A synthetic image and how to save it, as `MakeImage` in `compare/dump_tinyexr.cc` makes it.
struct ExrMade {
    header: tinyexr_nrp::ExrHeader,
    image: tinyexr_nrp::ExrImage,
    /// Whether what is saved of it has to be read back: not with attributes that say other
    /// than the image is, and not with a level left out.
    plain: bool,
}

/// The values of one channel, `w` by `h` of them, of one of four kinds: noise, a ramp, one
/// value, values such as a picture has. (`MakeValues`.)
fn exr_values(
    r: &mut Random,
    pixel_type: tinyexr_nrp::PixelType,
    w: i32,
    h: i32,
) -> tinyexr_nrp::ChannelData {
    use tinyexr_nrp::{ChannelData, PixelType};
    let kind = r.next() % 4;
    let first = r.next();
    let step_x = r.next() % 8;
    let step_y = r.next() % 8;
    let mut values = Vec::with_capacity((w * h) as usize);
    for y in 0..h as u32 {
        for x in 0..w as u32 {
            values.push(match kind {
                0 => r.next(),
                1 => first
                    .wrapping_add(x.wrapping_mul(step_x))
                    .wrapping_add(y.wrapping_mul(step_y)),
                2 => first,
                _ => {
                    let n = r.next();
                    if pixel_type == PixelType::FLOAT {
                        0x3c000000 + n % 0x04000000
                    } else if pixel_type == PixelType::HALF {
                        0x3000 + n % 0x1000
                    } else {
                        n % 1000
                    }
                }
            });
        }
    }
    if pixel_type == PixelType::FLOAT {
        ChannelData::Float(values.into_iter().map(f32::from_bits).collect())
    } else if pixel_type == PixelType::HALF {
        ChannelData::Half(values.into_iter().map(|value| value as u16).collect())
    } else {
        ChannelData::Uint(values)
    }
}

/// A double such as the settings of ZFP have: a rate, a tolerance, or one that is not either.
/// (`ZfpDouble`: less than nothing, which asks for most of four gigabits a block, only where
/// there is more than one channel, `many`.)
fn exr_zfp_double(r: &mut Random, tolerance: bool, many: bool) -> f64 {
    const RATES: [f64; 16] = [
        0.5, 1.0, 2.0, 2.0, 4.0, 7.3, 8.0, 12.0, 16.0, 24.0, 31.9, 32.0, 40.0, 0.0, -0.01, 1e-9,
    ];
    const TOLERANCES: [f64; 16] = [
        1e-3, 0.5, 1e-7, 0.01, 2.0, 1e-30, 0.0, -1.0, 1e-40, 1e30, 1e-320, 0.25, 3.0, 1e-5, 100.0,
        1e300,
    ];
    let kind = r.next() % 32;
    if kind < 16 {
        return if tolerance {
            TOLERANCES[kind as usize]
        } else {
            RATES[kind as usize]
        };
    }
    if kind < 28 {
        return f64::from(r.next() % 400) / if tolerance { 1000.0 } else { 10.0 };
    }
    match kind {
        28 => f64::NAN,
        29 => f64::INFINITY,
        30 if many => -1.0,
        _ => 100.0,
    }
}

/// A synthetic image (`MakeImage`); for ZFP compression (`g_zfp`), of floats, most of them of
/// whole blocks of four by four, with the attributes of ZFP's settings.
fn exr_image(r: &mut Random, small: bool, zfp: bool) -> ExrMade {
    use tinyexr_nrp::{
        self as tinyexr, Attribute, Box2i, ChannelInfo, ExrHeader, ExrImage, ExrTile,
    };
    const NAMES: [&str; 8] = ["A", "B", "G", "R", "Z", "id", "light.B", "light.G"];
    let mut plain = true;
    let tiled = r.next().is_multiple_of(3);
    let size_kind = if zfp {
        8 + r.next() % 8
    } else if small {
        0
    } else {
        r.next() % 8
    };
    let (width, height);
    if size_kind >= 8 {
        // (For ZFP: of whole blocks, now and then not.)
        if size_kind < 15 {
            width = 4 * (1 + (r.next() % 12) as i32);
            height = 4 * (1 + (r.next() % 12) as i32);
        } else {
            width = 1 + (r.next() % 40) as i32;
            height = 1 + (r.next() % 40) as i32;
        }
    } else if size_kind < 5 {
        width = 1 + (r.next() % if small { 10 } else { 40 }) as i32;
        height = 1 + (r.next() % if small { 36 } else { 40 }) as i32;
    } else if size_kind < 7 {
        width = 1 + (r.next() % 150) as i32;
        height = 1 + (r.next() % 150) as i32;
    } else {
        width = 1 + (r.next() % 400) as i32;
        height = 1 + (r.next() % 100) as i32;
    }

    // Its channels: red, green, blue and alpha; without alpha; one channel; or some of eight.
    let (count, start);
    let names_kind = r.next() % 4;
    if names_kind == 0 {
        count = 4;
        start = 0;
    } else if names_kind == 1 {
        count = 3;
        start = 1;
    } else if names_kind == 2 {
        count = 1;
        start = 0;
    } else {
        count = 1 + r.next() % 5;
        start = r.next() % (8 - count + 1);
    }
    let mut header = ExrHeader::default();
    for c in 0..count {
        let name = if names_kind == 2 {
            "Y"
        } else {
            NAMES[(start + c) as usize]
        };
        if zfp {
            // Floats, halves or integers handed over, stored as floats, now and then not.
            let kind = r.next() % 16;
            let memory_type = if kind < 10 {
                tinyexr::PixelType::FLOAT
            } else if kind < 15 {
                tinyexr::PixelType::HALF
            } else {
                tinyexr::PixelType::UINT
            };
            let as_half = r.next().is_multiple_of(24);
            let file_type = if as_half && memory_type != tinyexr::PixelType::UINT {
                tinyexr::PixelType::HALF
            } else {
                tinyexr::PixelType::FLOAT
            };
            header.channels.push(ChannelInfo {
                name: name.to_string(),
                pixel_type: file_type,
                x_sampling: 1,
                y_sampling: 1,
                p_linear: 0,
            });
            header.pixel_types.push(memory_type);
            header.requested_pixel_types.push(file_type);
            continue;
        }
        let memory_type = tinyexr::PixelType((r.next() % 3) as i32);
        let as_half = r.next() % 2;
        let file_type = if memory_type == tinyexr::PixelType::UINT {
            tinyexr::PixelType::UINT
        } else if as_half != 0 {
            tinyexr::PixelType::HALF
        } else {
            tinyexr::PixelType::FLOAT
        };
        header.channels.push(ChannelInfo {
            name: name.to_string(),
            pixel_type: file_type,
            x_sampling: 1,
            y_sampling: 1,
            p_linear: 0,
        });
        header.pixel_types.push(memory_type);
        header.requested_pixel_types.push(file_type);
    }

    // Its compression: one of the eight, PIZ more often; now and then one that is not written.
    let compression = r.next() % 20;
    header.compression_type = if compression < 16 {
        tinyexr_nrp::Compression((compression % 8) as i32)
    } else if compression == 16 {
        tinyexr::Compression::ZFP
    } else if compression == 17 {
        tinyexr_nrp::Compression(9)
    } else {
        tinyexr::Compression::PIZ
    };
    if zfp {
        header.compression_type = tinyexr::Compression::ZFP;
    } else if header.compression_type == tinyexr::Compression::ZFP {
        // (A ZFP file without the attributes of its settings, as these other images are, is
        // written, at 2 bits a value, and not read.)
        plain = false;
    }
    header.long_name = i32::from(r.next().is_multiple_of(8));
    if zfp {
        // The settings: a rate, a precision or a tolerance, each with its type; now and then
        // a type it does not know, none, a type of other than one byte, or no value for it.
        let kind = r.next() % 32;
        let zfp_type = if kind < 12 {
            0
        } else if kind < 18 {
            1
        } else if kind < 28 {
            2
        } else if kind == 28 {
            3 + r.next() % 253
        } else {
            0
        };
        let attribute = |name: &str, type_name: &str, value: Vec<u8>| Attribute {
            name: name.to_string(),
            type_name: type_name.to_string(),
            value,
        };
        if kind != 29 {
            header.custom_attributes.push(attribute(
                "zfpCompressionType",
                "uchar",
                vec![zfp_type as u8; if kind == 30 { 2 } else { 1 }],
            ));
        }
        if kind != 31 {
            if zfp_type == 0 {
                let rate = exr_zfp_double(r, false, header.channels.len() > 1);
                header.custom_attributes.push(attribute(
                    "zfpCompressionRate",
                    "double",
                    rate.to_le_bytes().to_vec(),
                ));
            } else if zfp_type == 1 {
                let precision = r.next() % 70;
                header.custom_attributes.push(attribute(
                    "zfpCompressionPrecision",
                    "int",
                    precision.to_le_bytes().to_vec(),
                ));
            } else {
                let tolerance = exr_zfp_double(r, true, header.channels.len() > 1);
                header.custom_attributes.push(attribute(
                    "zfpCompressionTolerance",
                    "double",
                    tolerance.to_le_bytes().to_vec(),
                ));
            }
        }
        // (A file without them saves, at 2 bits a value, and is not read.)
        if kind >= 28 {
            plain = false;
        }
    }
    // Now and then attributes of its own.
    let mut with_chunk_count = false;
    let mut wrong_chunk_count = 0u32;
    if !zfp && r.next().is_multiple_of(3) {
        let attributes = 1 + r.next() % 3;
        for j in 0..attributes {
            let kind = r.next() % 16;
            let mut attribute = Attribute {
                name: String::new(),
                type_name: "string".to_string(),
                value: Vec::new(),
            };
            if kind < 6 {
                attribute.name = if kind < 4 { "comment" } else { "owner" }.to_string();
                attribute.value = vec![0; (r.next() % 20) as usize];
                for byte in &mut attribute.value {
                    *byte = b'a' + (r.next() % 26) as u8;
                }
            } else if kind < 10 {
                const TYPES: [&str; 5] = [
                    "scanlineimage",
                    "tiledimage",
                    "deeptile",
                    "deepscanline",
                    "other",
                ];
                attribute.name = "type".to_string();
                attribute.value = TYPES[(r.next() % 5) as usize].as_bytes().to_vec();
                plain = false;
            } else if kind < 12 {
                attribute.name = "name".to_string();
                attribute.value = vec![0; 1 + (r.next() % 8) as usize];
                for byte in &mut attribute.value {
                    *byte = b'a' + (r.next() % 26) as u8;
                }
            } else if kind < 15 {
                // (Its value is filled in once the image is made.)
                let wrong = r.next() % 4;
                let count = r.next() % 40;
                if with_chunk_count {
                    continue;
                }
                with_chunk_count = true;
                wrong_chunk_count = if wrong == 0 { count + 1 } else { 0 };
                attribute.name = "chunkCount".to_string();
                attribute.type_name = "int".to_string();
                attribute.value = vec![0; 4];
            } else {
                let many = 126 + r.next() % 6;
                attribute.type_name = "int".to_string();
                attribute.value = vec![j as u8; 4];
                for k in 0..many {
                    attribute.name = format!("a{k:03}");
                    header.custom_attributes.push(attribute.clone());
                }
                continue;
            }
            header.custom_attributes.push(attribute);
        }
    }
    // The count of its chunks into the attribute for it, if it has one: the count, or another.
    let mut set_chunk_count = |header: &mut ExrHeader, count: u32| {
        for attribute in &mut header.custom_attributes {
            if attribute.name == "chunkCount" {
                let mut value = count;
                if wrong_chunk_count != 0 {
                    let other = wrong_chunk_count - 1;
                    value = if other >= count { other + 1 } else { other };
                }
                attribute.value = value.to_le_bytes().to_vec();
                if value != count {
                    plain = false;
                }
            }
        }
    };

    let mut image = ExrImage {
        width,
        height,
        num_channels: count as i32,
        ..ExrImage::default()
    };
    header.data_window = Box2i {
        min_x: 0,
        min_y: 0,
        max_x: width - 1,
        max_y: height - 1,
    };
    if !tiled {
        for c in 0..count as usize {
            image
                .images
                .push(exr_values(r, header.pixel_types[c], width, height));
        }
        let lines = match header.compression_type {
            tinyexr::Compression::ZIP | tinyexr::Compression::PXR24 | tinyexr::Compression::ZFP => {
                16
            }
            tinyexr::Compression::PIZ | tinyexr::Compression::B44 | tinyexr::Compression::B44A => {
                32
            }
            _ => 1,
        };
        set_chunk_count(&mut header, ((height + lines - 1) / lines) as u32);
        return ExrMade {
            header,
            image,
            plain,
        };
    }

    header.tiled = 1;
    header.tile_size_x = 1 + (r.next() % width.min(64) as u32) as i32;
    header.tile_size_y = 1 + (r.next() % height.min(64) as u32) as i32;
    if zfp && !r.next().is_multiple_of(8) {
        // (Of whole blocks, and of one level, most of them.)
        header.tile_size_x = (header.tile_size_x + 3) & !3;
        header.tile_size_y = (header.tile_size_y + 3) & !3;
    }
    header.tile_level_mode = if zfp && !r.next().is_multiple_of(4) {
        tinyexr::TileLevelMode::ONE_LEVEL
    } else {
        tinyexr_nrp::TileLevelMode((r.next() % 3) as i32)
    };
    header.tile_rounding_mode = tinyexr_nrp::TileRoundingMode((r.next() % 2) as i32);
    let without_last_level = r.next().is_multiple_of(16);

    // Its levels, as tinyexr counts them.
    let levels = tinyexr::tile_levels(&header).expect("the levels of a synthetic image");
    let mut images = Vec::new();
    let mut chunks = 0u32;
    for level in &levels {
        let mut level_image = ExrImage {
            level_x: level.level_x,
            level_y: level.level_y,
            width: level.width,
            height: level.height,
            num_channels: count as i32,
            ..ExrImage::default()
        };
        for t in 0..level.num_x_tiles * level.num_y_tiles {
            let (offset_x, offset_y) = (t % level.num_x_tiles, t / level.num_x_tiles);
            let mut tile = ExrTile {
                offset_x,
                offset_y,
                level_x: level.level_x,
                level_y: level.level_y,
                width: header
                    .tile_size_x
                    .min(level.width - offset_x * header.tile_size_x),
                height: header
                    .tile_size_y
                    .min(level.height - offset_y * header.tile_size_y),
                images: Vec::new(),
            };
            for c in 0..count as usize {
                tile.images.push(exr_values(
                    r,
                    header.pixel_types[c],
                    header.tile_size_x,
                    header.tile_size_y,
                ));
            }
            level_image.tiles.push(tile);
            chunks += 1;
        }
        images.push(level_image);
    }
    set_chunk_count(&mut header, chunks);
    if without_last_level && images.len() > 1 {
        images.pop();
        plain = false;
    }
    // The levels one behind the other.
    let mut chain: Option<Box<ExrImage>> = None;
    while let Some(mut level) = images.pop() {
        level.next_level = chain.take();
        chain = Some(Box::new(level));
    }
    ExrMade {
        header,
        image: *chain.expect("a level"),
        plain,
    }
}

/// The `tinyexr_write` dumps: synthetic images saved, as `compare/dump_tinyexr.cc` saves them
/// with tinyexr: a hash of each file's bytes, or of the message where none is saved. Each
/// file is read back besides, which the other program does not do: what was saved as it was
/// has to come back as it was.
pub fn tinyexr_write(name: &str, seed: u64, count: u32) -> String {
    use tinyexr_nrp::{self as tinyexr, ChannelData};
    let mut r = Random { state: seed };
    let mut hash = Hash::new();
    let small = name.ends_with("_small");
    let mut file_total = 0u64;
    let (mut saved, mut not_saved, mut short_way) = (0u32, 0u32, 0u32);
    let mut out = String::new();
    // The values of a channel as bits, to compare them.
    fn bits(values: &ChannelData, index: usize) -> u32 {
        match values {
            ChannelData::Uint(values) => values[index],
            ChannelData::Half(values) => u32::from(values[index]),
            ChannelData::Float(values) => values[index].to_bits(),
        }
    }
    for i in 0..count {
        let made = exr_image(&mut r, small, name.starts_with("tinyexr_zfp"));
        match tinyexr::save_image_as_tinyexr(&made.image, &made.header) {
            Err(error) => {
                not_saved += 1;
                // (What tinyexr gives as the size: nothing, or one of its error numbers.)
                let refusal = if error.code == tinyexr::ErrorCode::SERIALIZATION_FAILED {
                    0
                } else {
                    error.code.0
                };
                hash.add(&refusal.to_le_bytes());
                hash.add(error.message.as_bytes());
                hash.add(&[0]);
            }
            Ok(file) => {
                saved += 1;
                file_total += file.len() as u64;
                hash.add(&(file.len() as u32).to_le_bytes());
                hash.add(&file);

                if made.plain {
                    // Read back from the file as the format has it, which is tinyexr's own
                    // unless a PXR24 or B44 block did not come out smaller than its pixels.
                    let lossy_blocks = matches!(
                        made.header.compression_type,
                        tinyexr::Compression::PXR24
                            | tinyexr::Compression::B44
                            | tinyexr::Compression::B44A
                    );
                    let file = if lossy_blocks {
                        tinyexr::save_image(&made.image, &made.header).unwrap_or_else(|error| {
                            panic!("{name}: file {i} is not saved the other way: {error:?}")
                        })
                    } else {
                        file
                    };
                    let version = tinyexr::parse_version(&file).unwrap_or_else(|code| {
                        panic!("{name}: file {i} is not read back: {code:?}")
                    });
                    let mut header =
                        tinyexr::parse_header(&version, &file).unwrap_or_else(|error| {
                            panic!("{name}: file {i} is not read back: {error:?}")
                        });
                    let read = tinyexr::load_image(&mut header, &file).unwrap_or_else(|error| {
                        panic!("{name}: file {i} is not read back: {error:?}")
                    });
                    // A channel stored as the type it had, by a compression that loses
                    // nothing of that type, comes back bit for bit.
                    let comp = made.header.compression_type;
                    let (mut ours, mut theirs) = (Some(&made.image), Some(&read));
                    while let (Some(wrote), Some(got)) = (ours, theirs) {
                        assert_eq!(
                            (wrote.width, wrote.height, wrote.tiles.len()),
                            (got.width, got.height, got.tiles.len()),
                            "{name}: file {i}"
                        );
                        for c in 0..made.header.channels.len() {
                            let file_type = made.header.requested_pixel_types[c];
                            let kept = made.header.pixel_types[c] == file_type
                                && comp != tinyexr::Compression::ZFP
                                && !(comp == tinyexr::Compression::PXR24
                                    && file_type == tinyexr::PixelType::FLOAT)
                                && !((comp == tinyexr::Compression::B44
                                    || comp == tinyexr::Compression::B44A)
                                    && file_type == tinyexr::PixelType::HALF);
                            if !kept {
                                continue;
                            }
                            if wrote.tiles.is_empty() {
                                for index in 0..(wrote.width * wrote.height) as usize {
                                    assert_eq!(
                                        bits(&wrote.images[c], index),
                                        bits(&got.images[c], index),
                                        "{name}: file {i}, channel {c}, value {index}"
                                    );
                                }
                            }
                            for (tile, got_tile) in wrote.tiles.iter().zip(&got.tiles) {
                                assert_eq!(
                                    (tile.width, tile.height, tile.offset_x, tile.offset_y),
                                    (
                                        got_tile.width,
                                        got_tile.height,
                                        got_tile.offset_x,
                                        got_tile.offset_y
                                    ),
                                    "{name}: file {i}"
                                );
                                for y in 0..tile.height as usize {
                                    for x in 0..tile.width as usize {
                                        let index = y * made.header.tile_size_x as usize + x;
                                        assert_eq!(
                                            bits(&tile.images[c], index),
                                            bits(&got_tile.images[c], index),
                                            "{name}: file {i}, channel {c}, a tile's value {index}"
                                        );
                                    }
                                }
                            }
                        }
                        ours = wrote.next_level.as_deref();
                        theirs = got.next_level.as_deref();
                    }
                    assert!(
                        ours.is_none() && theirs.is_none(),
                        "{name}: file {i} comes back with other levels"
                    );
                }
            }
        }

        // Every fourth, floats saved the short way: one, three or four to a pixel (or two,
        // which it does not take), as halves or as floats.
        if i % 4 == 0 {
            const COMPONENTS: [i32; 8] = [2, 1, 1, 3, 3, 3, 4, 4];
            let components = COMPONENTS[(r.next() % 8) as usize];
            let width = 1 + (r.next() % 40) as i32;
            let height = 1 + (r.next() % 40) as i32;
            let as_halves = r.next() % 2 != 0;
            let ChannelData::Float(floats) = exr_values(
                &mut r,
                tinyexr::PixelType::FLOAT,
                width * components,
                height,
            ) else {
                unreachable!()
            };
            match tinyexr::save_floats(&floats, width, height, components, as_halves) {
                Ok(file) => {
                    short_way += 1;
                    file_total += file.len() as u64;
                    hash.add(&(file.len() as u32).to_le_bytes());
                    hash.add(&file);
                    // And back: the floats, or the halves of them.
                    let tinyexr::RgbaImage {
                        width: read_width,
                        height: read_height,
                        data: rgba,
                    } = tinyexr::load_rgba(&file).unwrap_or_else(|error| {
                        panic!("{name}: file {i}, saved the short way, is not read back: {error:?}")
                    });
                    assert_eq!((read_width, read_height), (width, height));
                    if !as_halves {
                        for pixel in 0..(width * height) as usize {
                            for k in 0..components as usize {
                                // (One channel comes back as all four.)
                                let wrote = floats[pixel * components as usize + k];
                                assert_eq!(
                                    wrote.to_bits(),
                                    rgba[pixel * 4 + k].to_bits(),
                                    "{name}: file {i}, saved the short way, pixel {pixel}"
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    hash.add(&error.code.0.to_le_bytes());
                    hash.add(error.message.as_bytes());
                    hash.add(&[0]);
                }
            }
        }

        if (i + 1) % 200 == 0 || i + 1 == count {
            out += &format!("{name} {} {:016x}\n", i + 1, hash.value);
        }
    }
    out + &format!(
        "{name} bytes of files: {file_total}; saved: {saved}, not saved: {not_saved}, saved the short way: {short_way}\n"
    )
}

/// A number into a hash, as `AddNumber` in `compare/dump_tinyexr.cc` puts it.
fn exr_add_number(hash: &mut Hash, value: impl Into<i32>) {
    let value: i32 = value.into();
    hash.add(&value.to_le_bytes());
}

/// A name with its end, or a mark in its place where it was not UTF-8 (`AddName`).
fn exr_add_name(hash: &mut Hash, text: &str) {
    if text.contains('\u{fffd}') {
        hash.add(&0xfffffffdu32.to_le_bytes());
    } else {
        hash.add(text.as_bytes());
        hash.add(&[0]);
    }
}

/// An error's number and message, or a mark where it has none (`AddMessage2`).
fn exr_add_error(hash: &mut Hash, code: impl Into<i32>, message: &str) {
    exr_add_number(hash, code);
    if message.is_empty() {
        hash.add(&0xfffffffeu32.to_le_bytes());
    } else {
        hash.add(message.as_bytes());
        hash.add(&[0]);
    }
}

/// A header, as `AddHeader` puts it.
fn exr_add_header(hash: &mut Hash, header: &tinyexr_nrp::ExrHeader) {
    hash.add(&header.pixel_aspect_ratio.to_bits().to_le_bytes());
    exr_add_number(hash, header.line_order);
    for window in [header.data_window, header.display_window] {
        exr_add_number(hash, window.min_x);
        exr_add_number(hash, window.min_y);
        exr_add_number(hash, window.max_x);
        exr_add_number(hash, window.max_y);
    }
    hash.add(&header.screen_window_center[0].to_bits().to_le_bytes());
    hash.add(&header.screen_window_center[1].to_bits().to_le_bytes());
    hash.add(&header.screen_window_width.to_bits().to_le_bytes());
    exr_add_number(hash, header.chunk_count);
    exr_add_number(hash, header.tiled);
    exr_add_number(hash, header.tile_size_x);
    exr_add_number(hash, header.tile_size_y);
    exr_add_number(hash, header.tile_level_mode);
    exr_add_number(hash, header.tile_rounding_mode);
    exr_add_number(hash, header.long_name);
    exr_add_number(hash, header.non_image);
    exr_add_number(hash, header.multipart);
    exr_add_number(hash, header.header_len as i32);
    exr_add_number(hash, header.custom_attributes.len() as i32);
    for attribute in &header.custom_attributes {
        exr_add_name(hash, &attribute.name);
        exr_add_name(hash, &attribute.type_name);
        exr_add_number(hash, attribute.value.len() as i32);
        hash.add(&attribute.value);
    }
    exr_add_number(hash, header.channels.len() as i32);
    for (c, channel) in header.channels.iter().enumerate() {
        exr_add_name(hash, &channel.name);
        exr_add_number(hash, channel.pixel_type);
        exr_add_number(hash, channel.x_sampling);
        exr_add_number(hash, channel.y_sampling);
        exr_add_number(hash, i32::from(channel.p_linear));
        exr_add_number(hash, header.pixel_types[c]);
        exr_add_number(hash, header.requested_pixel_types[c]);
    }
    exr_add_number(hash, header.compression_type);
    exr_add_name(hash, &header.name);
}

/// An image and its levels, as `AddImage` puts them.
fn exr_add_image(hash: &mut Hash, image: &tinyexr_nrp::ExrImage, header: &tinyexr_nrp::ExrHeader) {
    use tinyexr_nrp::ChannelData;
    fn add_values(hash: &mut Hash, values: &ChannelData) {
        match values {
            ChannelData::Uint(values) => values
                .iter()
                .for_each(|value| hash.add(&value.to_le_bytes())),
            ChannelData::Half(values) => values
                .iter()
                .for_each(|value| hash.add(&value.to_le_bytes())),
            ChannelData::Float(values) => values
                .iter()
                .for_each(|value| hash.add(&value.to_bits().to_le_bytes())),
        }
    }
    for &pixel_type in &header.pixel_types {
        exr_add_number(hash, pixel_type);
    }
    let mut next = Some(image);
    while let Some(level) = next {
        exr_add_number(hash, level.level_x);
        exr_add_number(hash, level.level_y);
        exr_add_number(hash, level.width);
        exr_add_number(hash, level.height);
        exr_add_number(hash, level.num_channels);
        exr_add_number(hash, level.tiles.len() as i32);
        if !level.tiles.is_empty() {
            for tile in &level.tiles {
                exr_add_number(hash, tile.offset_x);
                exr_add_number(hash, tile.offset_y);
                exr_add_number(hash, tile.level_x);
                exr_add_number(hash, tile.level_y);
                exr_add_number(hash, tile.width);
                exr_add_number(hash, tile.height);
                for values in &tile.images {
                    add_values(hash, values);
                }
            }
        } else {
            for values in &level.images {
                add_values(hash, values);
            }
        }
        next = level.next_level.as_deref();
    }
}

/// Into the hash, what came of a reading, as `AddOutcome` puts it: the flags (1 where the
/// original's answer depends on what is outside the file, which the port refuses with
/// `PAST_THE_END`; 2 where it stops, which the port refuses, with that or another error), or
/// 0 and the hash of what was read. `theirs` is what the other program found; the flags.
fn exr_add_outcome(hash: &mut Hash, end: ExrEnd, read: &Hash, theirs: u32) -> u32 {
    let refused = end == ExrEnd::PastTheEnd || (theirs == 2 && end == ExrEnd::Error);
    let flags = if refused {
        if theirs != 0 { theirs } else { 1 }
    } else {
        0
    };
    hash.add(&flags.to_le_bytes());
    if flags == 0 {
        hash.add(&read.value.to_le_bytes());
    }
    flags
}

/// How a reading ended.
#[derive(Clone, Copy, PartialEq)]
enum ExrEnd {
    /// With what was read, or with the file left.
    Read,
    /// With an error.
    Error,
    /// Refused with `PAST_THE_END`.
    PastTheEnd,
}

/// The reading of a file of many parts, as `ReadParts` does it.
fn exr_read_parts(hash: &mut Hash, bytes: &[u8], wish: u32) -> ExrEnd {
    use tinyexr_nrp::{self as tinyexr, PAST_THE_END};
    let version = match tinyexr::parse_version(bytes) {
        Err(code) => {
            exr_add_number(hash, code);
            return ExrEnd::Error;
        }
        Ok(version) => version,
    };
    exr_add_number(hash, 0);
    exr_add_number(hash, version.version);
    exr_add_number(hash, i32::from(version.tiled));
    exr_add_number(hash, i32::from(version.long_name));
    exr_add_number(hash, i32::from(version.non_image));
    exr_add_number(hash, i32::from(version.multipart));
    let mut headers = match tinyexr::parse_multipart_headers_as_tinyexr(&version, bytes) {
        Err(error) if error.message == PAST_THE_END => return ExrEnd::PastTheEnd,
        Err(error) => {
            exr_add_error(hash, error.code, &error.message);
            return ExrEnd::Error;
        }
        Ok(headers) => headers,
    };
    exr_add_error(hash, 0, "");
    exr_add_number(hash, headers.len() as i32);
    let mut left = headers.len() > 64;
    let mut chunks = 0i64;
    for header in &headers {
        exr_add_header(hash, header);
        left |= tinyexr_left(header);
        chunks += i64::from(header.chunk_count.max(0));
    }
    if left || chunks > 1 << 16 {
        hash.add(&0xfffffffbu32.to_le_bytes());
        return ExrEnd::Read;
    }
    for (i, header) in headers.iter_mut().enumerate() {
        if i < 32 && (wish >> i) & 1 != 0 {
            for c in 0..header.channels.len() {
                if header.pixel_types[c] == tinyexr::PixelType::HALF {
                    header.requested_pixel_types[c] = tinyexr::PixelType::FLOAT;
                }
            }
        }
    }
    match tinyexr::load_multipart_images_as_tinyexr(&mut headers, bytes) {
        Err(error) if error.message == PAST_THE_END => ExrEnd::PastTheEnd,
        Err(error) => {
            exr_add_error(hash, error.code, &error.message);
            ExrEnd::Error
        }
        Ok(images) => {
            exr_add_error(hash, 0, "");
            for (image, header) in images.iter().zip(&headers) {
                exr_add_image(hash, image, header);
            }
            ExrEnd::Read
        }
    }
}

/// The `tinyexr_parts` dump: files of many parts made from the seed and saved, as
/// `compare/dump_tinyexr.cc` saves them with tinyexr (the bytes, or the message); and the
/// files that program wrote to the file of files (each with what is to be read of it as
/// floats, and what came of it there), read as it reads them.
pub fn tinyexr_parts(name: &str, seed: u64, count: u32, file: &[u8]) -> String {
    use tinyexr_nrp as tinyexr;
    let mut r = Random { state: seed };
    let mut hash = Hash::new();
    let mut file_total = 0u64;
    let (mut saved, mut not_saved, mut past_end, mut stopped) = (0u32, 0u32, 0u32, 0u32);
    let mut out = String::new();
    let mut at = 0;
    let number = |at: &mut usize| {
        let value = u32::from_le_bytes(file[*at..*at + 4].try_into().unwrap());
        *at += 4;
        value
    };
    for i in 0..count {
        let num_parts = if r.next().is_multiple_of(10) {
            1
        } else {
            2 + r.next() % 3
        };
        let mut parts = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for p in 0..num_parts {
            parts.push(exr_image(&mut r, true, false));
            let kind = r.next() % 40;
            let part_name = if kind == 0 {
                String::new()
            } else if kind == 1 && p > 0 {
                names[names.len() - 1].clone()
            } else {
                format!("part{p}")
            };
            names.push(part_name);
        }
        for (made, part_name) in parts.iter_mut().zip(&names) {
            made.header.name = part_name.clone();
        }
        let list: Vec<_> = parts
            .iter()
            .map(|made| (&made.image, &made.header))
            .collect();
        match tinyexr::save_multipart_images_as_tinyexr(&list) {
            Err(error) => {
                not_saved += 1;
                let refusal = if error.code == tinyexr::ErrorCode::SERIALIZATION_FAILED {
                    0
                } else {
                    error.code.0
                };
                hash.add(&refusal.to_le_bytes());
                hash.add(error.message.as_bytes());
                hash.add(&[0]);
            }
            Ok(saved_file) => {
                saved += 1;
                file_total += saved_file.len() as u64;
                hash.add(&(saved_file.len() as u32).to_le_bytes());
                hash.add(&saved_file);
            }
        }

        let wish = number(&mut at);
        let theirs = number(&mut at);
        let size = number(&mut at) as usize;
        let bytes = &file[at..at + size];
        at += size;
        let mut read = Hash::new();
        let end = exr_read_parts(&mut read, bytes, wish);
        let flags = exr_add_outcome(&mut hash, end, &read, theirs);
        past_end += u32::from(flags == 1);
        stopped += u32::from(flags == 2);

        if std::env::var_os("SHOW_EACH").is_some() {
            eprintln!("file {i} {:016x}", hash.value);
        }
        if (i + 1) % 200 == 0 || i + 1 == count {
            out += &format!("{name} {} {:016x}\n", i + 1, hash.value);
        }
    }
    out + &format!(
        "{name} bytes of files: {file_total}; saved: {saved}, not saved: {not_saved}; read where it depends on what is past the end: {past_end}, where tinyexr stops: {stopped}\n"
    )
}

/// The `tinyexr_deep` dump: the deep files `compare/dump_tinyexr.cc` made and wrote to the
/// file of files (each with what came of it there), read as it reads them.
pub fn tinyexr_deep(name: &str, count: u32, file: &[u8]) -> String {
    use tinyexr_nrp::{self as tinyexr, PAST_THE_END};
    let mut hash = Hash::new();
    let mut file_total = 0u64;
    let (mut past_end, mut stopped, mut left) = (0u32, 0u32, 0u32);
    let mut out = String::new();
    let mut at = 0;
    let number = |at: &mut usize| {
        let value = u32::from_le_bytes(file[*at..*at + 4].try_into().unwrap());
        *at += 4;
        value
    };
    for i in 0..count {
        let theirs = number(&mut at);
        let size = number(&mut at) as usize;
        let bytes = &file[at..at + size];
        at += size;
        file_total += size as u64;
        exr_add_number(&mut hash, size as i32);
        hash.add(bytes);

        if theirs == 4 {
            // (Left unread there: more pixels than there is time and memory for.)
            left += 1;
            hash.add(&4u32.to_le_bytes());
        } else {
            let mut read = Hash::new();
            let end = match tinyexr::load_deep(bytes) {
                Err(error) if error.message == PAST_THE_END => ExrEnd::PastTheEnd,
                Err(error) => {
                    exr_add_error(&mut read, error.code, &error.message);
                    ExrEnd::Error
                }
                Ok(deep) => {
                    exr_add_error(&mut read, 0, "");
                    exr_add_number(&mut read, deep.width);
                    exr_add_number(&mut read, deep.height);
                    exr_add_number(&mut read, deep.num_channels);
                    for channel_name in &deep.channel_names {
                        exr_add_name(&mut read, channel_name);
                    }
                    if deep.width > 0 && deep.height > 0 {
                        for row in &deep.offset_table {
                            for &offset in row {
                                exr_add_number(&mut read, offset);
                            }
                        }
                        for rows in &deep.image {
                            for (y, row) in rows.iter().enumerate() {
                                exr_add_number(&mut read, y as i32);
                                let n = deep.offset_table[y][deep.width as usize - 1];
                                if n as usize != row.len() {
                                    read.add(b"a line of other than its count of samples");
                                }
                                for value in row {
                                    read.add(&value.to_bits().to_le_bytes());
                                }
                            }
                        }
                    }
                    ExrEnd::Read
                }
            };
            let flags = exr_add_outcome(&mut hash, end, &read, theirs);
            past_end += u32::from(flags == 1);
            stopped += u32::from(flags == 2);
        }

        if std::env::var_os("SHOW_EACH").is_some() {
            eprintln!("file {i} {:016x}", hash.value);
        }
        if (i + 1) % 200 == 0 || i + 1 == count {
            out += &format!("{name} {} {:016x}\n", i + 1, hash.value);
        }
    }
    out + &format!(
        "{name} bytes of files: {file_total}; read where it depends on what is outside the file: {past_end}, where tinyexr stops: {stopped}, left: {left}\n"
    )
}
