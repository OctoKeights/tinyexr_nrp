//! Files of many parts: a port of `ParseEXRMultipartHeaderFromMemory` and
//! `LoadEXRMultipartImageFromMemory` of tinyexr v3.4.0 (`tinyexr.h`). (The writer is
//! [`super::save_multipart_images`].)
//!
//! Both read past the end of a damaged file: the first parses every header as if the whole
//! of the file after the first one were left, the second reads its table of offsets, and the
//! number of the part before each block, without a look at where the file ends. Here those
//! reads are made as the original makes them, with what is past the end taken to be all
//! zeros and then all ones. Where the answer is the same either way it is the original's
//! answer whatever its memory holds, and it is given; where it is not, the file is refused
//! with [`PAST_THE_END`].

use super::chunk::{OffsetData, decode_chunk, precalculate_tile_info, tile_offsets_shape};
use super::{
    ErrorCode, ExrError, ExrHeader, ExrImage, ExrVersion, HeaderInfo, Parsed, TileLevelMode,
    convert_header, parse_exr_header,
};

const EXR_VERSION_SIZE: usize = 8;

/// The message of a file refused because the original's answer for it depends on memory past
/// its end, or because the original stops on it. (Its code is `ErrorCode`.)
pub const PAST_THE_END: &str =
    "The file is damaged: tinyexr reads past its end for it, or stops on it.";

fn past_the_end() -> ExrError {
    ExrError {
        code: ErrorCode::INVALID_DATA,
        message: PAST_THE_END.to_string(),
    }
}

/// `ParseEXRMultipartHeaderFromMemory`: the header of every part of a file of many parts.
///
/// # Errors
///
/// A header that is damaged or lacks a required attribute, with tinyexr's error code and message.
pub fn parse_multipart_headers(
    version: ExrVersion,
    memory: &[u8],
) -> Result<Vec<ExrHeader>, ExrError> {
    parse_multipart_header(version, memory, false)
}

/// `ParseEXRMultipartHeaderFromMemory` exactly as tinyexr has it, which refuses DWAA and DWAB.
///
/// # Errors
///
/// As [`parse_multipart_headers`], and DWAA or DWAB compression.
pub fn parse_multipart_headers_as_tinyexr(
    version: ExrVersion,
    memory: &[u8],
) -> Result<Vec<ExrHeader>, ExrError> {
    parse_multipart_header(version, memory, true)
}

fn parse_multipart_header(
    version: ExrVersion,
    memory: &[u8],
    as_tinyexr: bool,
) -> Result<Vec<ExrHeader>, ExrError> {
    if memory.len() < EXR_VERSION_SIZE {
        return Err(ExrError {
            code: ErrorCode::INVALID_DATA,
            message: "Data size too short".to_string(),
        });
    }
    // (Every header but the first is read as if it had the whole of the file after the first
    // header to itself: the second time round, with ones past the end.)
    let (zeros, beyond) = parse_headers(version, memory, 0x00, as_tinyexr);
    if !beyond {
        return zeros;
    }
    let (ones, _) = parse_headers(version, memory, 0xff, as_tinyexr);
    if same(&zeros, &ones) {
        zeros
    } else {
        Err(past_the_end())
    }
}

/// Whether two answers are the same, their floats bit for bit.
fn same(a: &Result<Vec<ExrHeader>, ExrError>, b: &Result<Vec<ExrHeader>, ExrError>) -> bool {
    // (The floats compared as bits, and the rest with them set aside.)
    fn floats(header: &ExrHeader) -> [u32; 4] {
        [
            header.pixel_aspect_ratio.to_bits(),
            header.screen_window_center[0].to_bits(),
            header.screen_window_center[1].to_bits(),
            header.screen_window_width.to_bits(),
        ]
    }
    fn without_floats(header: &ExrHeader) -> ExrHeader {
        ExrHeader {
            pixel_aspect_ratio: 0.0,
            screen_window_center: [0.0; 2],
            screen_window_width: 0.0,
            ..header.clone()
        }
    }
    match (a, b) {
        (Ok(a), Ok(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| floats(a) == floats(b) && without_floats(a) == without_floats(b))
        }
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

/// The headers as the original parses them, with `pad` past the end of the file; and
/// whether the parsing was handed bytes past the end.
fn parse_headers(
    version: ExrVersion,
    memory: &[u8],
    pad: u8,
    as_tinyexr: bool,
) -> (Result<Vec<ExrHeader>, ExrError>, bool) {
    let size = memory.len();
    let marker_size = size - EXR_VERSION_SIZE;
    let mut padded = memory.to_vec();
    let mut marker = EXR_VERSION_SIZE;
    let mut beyond = false;

    let mut infos: Vec<HeaderInfo> = Vec::new();
    loop {
        let mut info = HeaderInfo::default();

        if marker + marker_size > padded.len() {
            padded.resize(marker + marker_size, pad);
        }
        beyond |= marker + marker_size > size;

        match parse_exr_header(
            &mut info,
            version,
            &padded[marker..marker + marker_size],
            as_tinyexr,
        ) {
            Ok(Parsed::Header) => {}
            Ok(Parsed::EndOfHeaders) => break,
            Err(error) => return (Err(error), beyond),
        }

        // `chunkCount` must exist in the header.
        if info.chunk_count == 0 {
            return (
                Err(ExrError {
                    code: ErrorCode::INVALID_DATA,
                    message: "`chunkCount' attribute is not found in the header.".to_string(),
                }),
                beyond,
            );
        }

        // move to next header.
        marker += info.header_len as usize;
        infos.push(info);
    }

    let mut headers = Vec::with_capacity(infos.len());
    // (Every header is converted; the last that fails is the one given.)
    let mut failure = None;
    for info in &infos {
        let mut exr_header = convert_header(info).unwrap_or_else(|message| {
            failure = Some(message);
            ExrHeader::default()
        });
        exr_header.multipart = version.multipart;
        headers.push(exr_header);
    }
    if let Some(message) = failure {
        return (
            Err(ExrError::new(ErrorCode::INVALID_HEADER, message)),
            beyond,
        );
    }
    (Ok(headers), beyond)
}

/// What a read of bytes past the end of the file makes of the rest: with zeros past the end
/// the original goes on, with ones it stops with this error. (Ones past the end make every
/// such offset and part number too large, and zeros may not.)
struct Divergence {
    error: ExrError,
}

/// A little-endian number of `N` bytes at `at`, as the original reads it whatever the end of
/// the file: with zeros past the end, and whether any byte was past it.
fn read_le<const N: usize>(memory: &[u8], at: usize) -> (u64, bool) {
    let mut value = 0u64;
    let mut beyond = false;
    for k in (0..N).rev() {
        let byte = at
            .checked_add(k)
            .and_then(|i| memory.get(i))
            .copied()
            .unwrap_or_else(|| {
                beyond = true;
                0
            });
        value = value << 8 | u64::from(byte);
    }
    (value, beyond)
}

/// `LoadEXRMultipartImageFromMemory`: the pixels of every part of the file, whose headers
/// [`parse_multipart_headers`] gave. Each part's channels come as its
/// header's `requested_pixel_types` ask, as [`super::load_image`] gives them.
///
/// # Errors
///
/// Headers not given by [`parse_multipart_headers`], or pixels that are damaged, with tinyexr's
/// error code and message.
pub fn load_multipart_images(
    exr_headers: &mut [ExrHeader],
    memory: &[u8],
) -> Result<Vec<ExrImage>, ExrError> {
    load_parts(exr_headers, memory, false)
}

/// `LoadEXRMultipartImageFromMemory` exactly as tinyexr has it
/// ([`super::load_image_as_tinyexr`] says where that differs).
///
/// # Errors
///
/// As [`load_multipart_images`].
pub fn load_multipart_images_as_tinyexr(
    exr_headers: &mut [ExrHeader],
    memory: &[u8],
) -> Result<Vec<ExrImage>, ExrError> {
    load_parts(exr_headers, memory, true)
}

fn load_parts(
    exr_headers: &mut [ExrHeader],
    memory: &[u8],
    as_tinyexr: bool,
) -> Result<Vec<ExrImage>, ExrError> {
    let headers_before = exr_headers.to_vec();
    let mut divergence: Option<Divergence> = None;
    let result = load_parts_with_zeros(exr_headers, memory, as_tinyexr, &mut divergence);
    match divergence {
        None => result,
        // With ones past the end the original stopped with an error there; with zeros it went
        // on to `result`. The same either way, or not.
        Some(Divergence { error }) => match result {
            Err(ref zeros) if *zeros == error => result,
            _ => {
                exr_headers.clone_from_slice(&headers_before);
                Err(past_the_end())
            }
        },
    }
}

/// The original, with zeros past the end of the file: and where it first read past the end in
/// a way ones there would have stopped it, how.
fn load_parts_with_zeros(
    exr_headers: &mut [ExrHeader],
    memory: &[u8],
    as_tinyexr: bool,
    divergence: &mut Option<Divergence>,
) -> Result<Vec<ExrImage>, ExrError> {
    fn fail<T>(message: &str) -> Result<T, ExrError> {
        Err(ExrError {
            code: ErrorCode::INVALID_DATA,
            message: message.to_string(),
        })
    }
    let size = memory.len();
    let num_parts = exr_headers.len();
    if num_parts == 0 || size <= EXR_VERSION_SIZE {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: "Invalid argument for LoadEXRMultipartImageFromMemory()".to_string(),
        });
    }

    // compute total header size.
    let mut total_header_size = 0usize;
    for exr_header in exr_headers.iter() {
        if exr_header.header_len == 0 {
            return Err(ExrError {
                code: ErrorCode::INVALID_ARGUMENT,
                message: "EXRHeader variable is not initialized.".to_string(),
            });
        }
        total_header_size += exr_header.header_len as usize;
    }

    let mut marker = total_header_size + 4 + 4; // +8 for magic number and version header.
    marker += 1; // Skip empty header.

    // NOTE 1:
    //   In multipart image, There is 'part number' before chunk data.
    //   4 byte : part number
    //   4+     : chunk
    //
    // NOTE 2:
    //   EXR spec says 'part number' is 'unsigned long' but actually this is
    //   'unsigned int(4 bytes)' in OpenEXR implementation...
    //   http://www.openexr.com/openexrfilelayout.pdf

    // One entry of a table of offsets, and past the end of the file what zeros there make of
    // it. (Once past the end with zeros, nothing the original goes on to do can end in the
    // error ones there end in: the offsets read as 0 from then on, which it takes.)
    let read_offset = |marker: &mut usize| -> Result<u64, ExrError> {
        let (offset, beyond) = read_le::<8>(memory, *marker);
        if offset >= size as u64 {
            return fail("Invalid offset size in EXR header chunks.");
        }
        if beyond {
            return Err(past_the_end());
        }
        *marker += 8;
        Ok(offset + 4) // +4 to skip 'part number'
    };

    // Load chunk offset table.
    let mut chunk_offset_table_list: Vec<OffsetData> = Vec::with_capacity(num_parts);
    for exr_header in exr_headers.iter() {
        let mut offset_data = OffsetData::default();
        if !exr_header.tiled || exr_header.tile_level_mode == TileLevelMode::ONE_LEVEL {
            // `InitSingleResolutionOffsets`. (A count below zero is one the original cannot
            // make a table of, which stops it.)
            if exr_header.chunk_count < 0 {
                return Err(past_the_end());
            }
            let mut offsets = Vec::new();
            for _ in 0..exr_header.chunk_count {
                offsets.push(read_offset(&mut marker)?);
            }
            offset_data.offsets = vec![vec![offsets]];
            offset_data.num_x_levels = 1;
            offset_data.num_y_levels = 1;
        } else {
            let Some((num_x_tiles, num_y_tiles)) = precalculate_tile_info(exr_header) else {
                return fail("Invalid tile info.");
            };
            let (shape, num_blocks) =
                tile_offsets_shape(&mut offset_data, exr_header, &num_x_tiles, &num_y_tiles);
            if num_blocks != exr_header.chunk_count {
                return fail("Invalid offset table size.");
            }
            for &(rows, columns) in &shape {
                let mut level = Vec::with_capacity(rows);
                for _ in 0..rows {
                    let mut row = Vec::with_capacity(columns);
                    for _ in 0..columns {
                        row.push(read_offset(&mut marker)?);
                    }
                    level.push(row);
                }
                offset_data.offsets.push(level);
            }
        }
        chunk_offset_table_list.push(offset_data);
    }

    // Decode image.
    let mut exr_images = Vec::with_capacity(num_parts);
    for (i, offset_data) in chunk_offset_table_list.iter().enumerate() {
        // First check 'part number' is identical to 'i'
        for &offset in offset_data.offsets.iter().flatten().flatten() {
            // -4 to move to 'part number' field.
            let (part_no, beyond) = read_le::<4>(memory, offset as usize - 4);
            if part_no != i as u64 {
                return fail("Invalid `part number' in EXR header chunks.");
            }
            if beyond && divergence.is_none() {
                // (With ones past the end, the number is never `i`.)
                *divergence = Some(Divergence {
                    error: ExrError {
                        code: ErrorCode::INVALID_DATA,
                        message: "Invalid `part number' in EXR header chunks.".to_string(),
                    },
                });
            }
        }

        exr_images.push(decode_chunk(
            &mut exr_headers[i],
            offset_data,
            memory,
            as_tinyexr,
        )?);
    }

    Ok(exr_images)
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChannelData, ChannelInfo, Compression, ExrHeader, ExrImage, PixelType,
        load_multipart_images, parse_multipart_headers, parse_version, save_multipart_images,
    };

    fn part(name: &str, width: i32, height: i32, value: f32) -> (ExrImage, ExrHeader) {
        let image = ExrImage {
            images: vec![ChannelData::Float(vec![value; (width * height) as usize])],
            width,
            height,
            num_channels: 1,
            ..ExrImage::default()
        };
        let header = ExrHeader {
            channels: vec![ChannelInfo {
                name: "Y".to_string(),
                pixel_type: PixelType::HALF,
                x_sampling: 1,
                y_sampling: 1,
                p_linear: 0,
            }],
            pixel_types: vec![PixelType::FLOAT],
            requested_pixel_types: vec![PixelType::HALF],
            compression_type: Compression::ZIP,
            name: name.to_string(),
            ..ExrHeader::default()
        };
        (image, header)
    }

    #[test]
    fn parts_are_written_and_read_back() {
        let (a, a_header) = part("left", 20, 17, 0.5);
        let (b, b_header) = part("right", 3, 40, 2.0);
        let file = save_multipart_images(&[(&a, &a_header), (&b, &b_header)]).unwrap();
        let version = parse_version(&file).unwrap();
        assert!(version.multipart);
        let mut headers = parse_multipart_headers(version, &file).unwrap();
        assert_eq!(
            headers
                .iter()
                .map(|header| header.name.as_str())
                .collect::<Vec<_>>(),
            ["left", "right"]
        );
        let images = load_multipart_images(&mut headers, &file).unwrap();
        assert_eq!(
            (
                images[0].width,
                images[0].height,
                images[1].width,
                images[1].height
            ),
            (20, 17, 3, 40)
        );
        assert_eq!(images[1].images[0], ChannelData::Half(vec![0x4000; 120]));
    }

    #[test]
    fn parts_need_names_of_their_own() {
        let (a, a_header) = part("same", 2, 2, 1.0);
        let (b, b_header) = part("same", 2, 2, 1.0);
        let error = save_multipart_images(&[(&a, &a_header), (&b, &b_header)]).unwrap_err();
        assert_eq!(
            error.message,
            "'name' attributes must be unique for a multi-part file"
        );
        let (c, mut c_header) = part("", 2, 2, 1.0);
        c_header.name.clear();
        let error = save_multipart_images(&[(&a, &a_header), (&c, &c_header)]).unwrap_err();
        assert_eq!(
            error.message,
            "Invalid 'name' attribute for a multi-part file"
        );
    }

    #[test]
    fn a_file_cut_short_is_refused_where_the_original_reads_past_its_end() {
        let (a, a_header) = part("left", 20, 17, 0.5);
        let (b, b_header) = part("right", 3, 40, 2.0);
        let file = save_multipart_images(&[(&a, &a_header), (&b, &b_header)]).unwrap();
        let version = parse_version(&file).unwrap();
        let mut headers = parse_multipart_headers(version, &file).unwrap();
        // Cut in the table of offsets: the original reads the rest of it past the end.
        let end_of_headers = 8
            + headers
                .iter()
                .map(|header| header.header_len as usize)
                .sum::<usize>()
            + 1;
        let cut = &file[..end_of_headers + 12];
        assert!(load_multipart_images(&mut headers, cut).is_err());
    }
}
