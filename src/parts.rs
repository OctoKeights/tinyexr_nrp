//! Files of many parts: a port of `ParseEXRMultipartHeaderFromMemory` and
//! `LoadEXRMultipartImageFromMemory` of tinyexr v3.4.0 (`tinyexr.h`), plus selective
//! single-part decoding ([`load_multipart_part`]). (The writer is
//! [`super::save_multipart_images`].)

use alloc::{string::ToString, vec, vec::Vec};

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

        if info.chunk_count == 0 {
            return (
                Err(ExrError {
                    code: ErrorCode::INVALID_DATA,
                    message: "`chunkCount' attribute is not found in the header.".to_string(),
                }),
                beyond,
            );
        }

        marker += info.header_len as usize;
        infos.push(info);
    }

    let mut headers = Vec::with_capacity(infos.len());
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

/// What a read of bytes past the end of the file makes of the rest.
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

/// Decodes only the single part at `part_index` (`0..exr_headers.len()`) from a multipart
/// `.exr` file in `memory`, without allocating or decompressing the other parts.
///
/// Updates `exr_headers[part_index].pixel_types` to `requested_pixel_types`, leaving all
/// other headers in `exr_headers` unchanged.
///
/// # Errors
///
/// Returns [`ErrorCode::INVALID_ARGUMENT`] if `exr_headers` is empty, `part_index` is out of
/// bounds, or a header is uninitialized, or [`ExrError`] if the offset tables or target
/// part's chunks are damaged.
pub fn load_multipart_part(
    exr_headers: &mut [ExrHeader],
    part_index: usize,
    memory: &[u8],
) -> Result<ExrImage, ExrError> {
    if part_index >= exr_headers.len() {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: "Part index out of range for LoadEXRMultipartPart".to_string(),
        });
    }
    let header_before = exr_headers[part_index].clone();
    let mut divergence: Option<Divergence> = None;
    let result =
        load_single_part_with_zeros(exr_headers, part_index, memory, false, &mut divergence);
    match divergence {
        None => result,
        Some(Divergence { error }) => match result {
            Err(ref zeros) if *zeros == error => result,
            _ => {
                exr_headers[part_index] = header_before;
                Err(past_the_end())
            }
        },
    }
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
        Some(Divergence { error }) => match result {
            Err(ref zeros) if *zeros == error => result,
            _ => {
                exr_headers.clone_from_slice(&headers_before);
                Err(past_the_end())
            }
        },
    }
}

/// Validates and advances `marker` past `exr_header`'s offset table, returning `Some(OffsetData)`
/// when `collect_offsets` is `true` and `None` when skipping non-selected parts.
fn read_part_offset_table(
    exr_header: &ExrHeader,
    memory: &[u8],
    marker: &mut usize,
    collect_offsets: bool,
) -> Result<Option<OffsetData>, ExrError> {
    fn fail<T>(message: &str) -> Result<T, ExrError> {
        Err(ExrError {
            code: ErrorCode::INVALID_DATA,
            message: message.to_string(),
        })
    }
    let size = memory.len();
    let read_offset = |marker: &mut usize| -> Result<u64, ExrError> {
        let (offset, beyond) = read_le::<8>(memory, *marker);
        if offset >= size as u64 {
            return fail("Invalid offset size in EXR header chunks.");
        }
        if beyond {
            return Err(past_the_end());
        }
        *marker = marker.checked_add(8).ok_or_else(past_the_end)?;
        Ok(offset + 4) // +4 to skip 'part number'
    };

    let mut offset_data = OffsetData::default();
    if !exr_header.tiled || exr_header.tile_level_mode == TileLevelMode::ONE_LEVEL {
        if exr_header.chunk_count < 0 {
            return Err(past_the_end());
        }
        let count = exr_header.chunk_count as usize;
        if collect_offsets {
            let cap = count.min(size.saturating_sub(*marker) / 8);
            let mut offsets = Vec::with_capacity(cap);
            for _ in 0..count {
                offsets.push(read_offset(marker)?);
            }
            offset_data.offsets = vec![vec![offsets]];
            offset_data.num_x_levels = 1;
            offset_data.num_y_levels = 1;
            Ok(Some(offset_data))
        } else {
            for _ in 0..count {
                read_offset(marker)?;
            }
            Ok(None)
        }
    } else {
        let Some((num_x_tiles, num_y_tiles)) = precalculate_tile_info(exr_header) else {
            return fail("Invalid tile info.");
        };
        let (shape, num_blocks) =
            tile_offsets_shape(&mut offset_data, exr_header, &num_x_tiles, &num_y_tiles);
        if num_blocks != exr_header.chunk_count {
            return fail("Invalid offset table size.");
        }
        if collect_offsets {
            // Cap pre-allocation to what the remaining file bytes can hold.
            let max_rem = size.saturating_sub(*marker) / 8;
            offset_data.offsets = Vec::with_capacity(shape.len().min(max_rem));
            for &(rows, columns) in &shape {
                let row_cap = rows.min(size.saturating_sub(*marker) / 8).min(1024);
                let mut level = Vec::with_capacity(row_cap);
                for _ in 0..rows {
                    let col_cap = columns.min(size.saturating_sub(*marker) / 8);
                    let mut row = Vec::with_capacity(col_cap);
                    for _ in 0..columns {
                        row.push(read_offset(marker)?);
                    }
                    level.push(row);
                }
                offset_data.offsets.push(level);
            }
            Ok(Some(offset_data))
        } else {
            for &(rows, columns) in &shape {
                for _ in 0..rows {
                    for _ in 0..columns {
                        read_offset(marker)?;
                    }
                }
            }
            Ok(None)
        }
    }
}

fn compute_offset_table_start(exr_headers: &[ExrHeader], size: usize) -> Result<usize, ExrError> {
    if exr_headers.is_empty() || size <= EXR_VERSION_SIZE {
        return Err(ExrError {
            code: ErrorCode::INVALID_ARGUMENT,
            message: "Invalid argument for LoadEXRMultipartImageFromMemory()".to_string(),
        });
    }
    let mut total_header_size = 0usize;
    for exr_header in exr_headers {
        if exr_header.header_len == 0 {
            return Err(ExrError {
                code: ErrorCode::INVALID_ARGUMENT,
                message: "EXRHeader variable is not initialized.".to_string(),
            });
        }
        total_header_size = total_header_size
            .checked_add(exr_header.header_len as usize)
            .ok_or_else(past_the_end)?;
    }
    // +8 for magic number and version header, +1 to skip empty header terminator.
    let marker = total_header_size.checked_add(9).ok_or_else(past_the_end)?;
    if marker > size {
        return Err(past_the_end());
    }
    Ok(marker)
}

fn validate_part_numbers(
    part_index: usize,
    offset_data: &OffsetData,
    memory: &[u8],
    divergence: &mut Option<Divergence>,
) -> Result<(), ExrError> {
    for &offset in offset_data.offsets.iter().flatten().flatten() {
        let (part_no, beyond) = read_le::<4>(memory, offset as usize - 4);
        if part_no != part_index as u64 {
            return Err(ExrError {
                code: ErrorCode::INVALID_DATA,
                message: "Invalid `part number' in EXR header chunks.".to_string(),
            });
        }
        if beyond && divergence.is_none() {
            *divergence = Some(Divergence {
                error: ExrError {
                    code: ErrorCode::INVALID_DATA,
                    message: "Invalid `part number' in EXR header chunks.".to_string(),
                },
            });
        }
    }
    Ok(())
}

fn load_parts_with_zeros(
    exr_headers: &mut [ExrHeader],
    memory: &[u8],
    as_tinyexr: bool,
    divergence: &mut Option<Divergence>,
) -> Result<Vec<ExrImage>, ExrError> {
    let mut marker = compute_offset_table_start(exr_headers, memory.len())?;
    let num_parts = exr_headers.len();

    let mut chunk_offset_table_list: Vec<OffsetData> = Vec::with_capacity(num_parts.min(1024));
    for exr_header in exr_headers.iter() {
        let offset_data = read_part_offset_table(exr_header, memory, &mut marker, true)?
            .expect("collected offsets");
        chunk_offset_table_list.push(offset_data);
    }

    let mut exr_images = Vec::with_capacity(num_parts.min(1024));
    for (i, offset_data) in chunk_offset_table_list.iter().enumerate() {
        validate_part_numbers(i, offset_data, memory, divergence)?;
        exr_images.push(decode_chunk(
            &mut exr_headers[i],
            offset_data,
            memory,
            as_tinyexr,
        )?);
    }

    Ok(exr_images)
}

fn load_single_part_with_zeros(
    exr_headers: &mut [ExrHeader],
    part_index: usize,
    memory: &[u8],
    as_tinyexr: bool,
    divergence: &mut Option<Divergence>,
) -> Result<ExrImage, ExrError> {
    let mut marker = compute_offset_table_start(exr_headers, memory.len())?;

    let mut selected_offsets: Option<OffsetData> = None;
    for (i, exr_header) in exr_headers.iter().enumerate() {
        let got = read_part_offset_table(exr_header, memory, &mut marker, i == part_index)?;
        if i == part_index {
            selected_offsets = got;
        }
    }

    let offset_data = selected_offsets.expect("part_index < exr_headers.len()");
    validate_part_numbers(part_index, &offset_data, memory, divergence)?;
    decode_chunk(
        &mut exr_headers[part_index],
        &offset_data,
        memory,
        as_tinyexr,
    )
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChannelData, ChannelInfo, Compression, ErrorCode, ExrHeader, ExrImage, PixelType,
        load_multipart_images, load_multipart_part, parse_multipart_headers, parse_version,
        save_multipart_images,
    };
    use alloc::{string::ToString, vec, vec::Vec};

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

        // Selective single-part decode of part 1 only:
        let mut selective_headers = headers.clone();
        selective_headers[1].requested_pixel_types[0] = PixelType::FLOAT;
        let only_right = load_multipart_part(&mut selective_headers, 1, &file).unwrap();
        assert_eq!((only_right.width, only_right.height), (3, 40));
        assert_eq!(only_right.images[0], ChannelData::Float(vec![2.0; 120]));
        // Part 0's header was left untouched (`pixel_types` stayed HALF).
        assert_eq!(selective_headers[0].pixel_types, vec![PixelType::HALF]);
        assert_eq!(selective_headers[1].pixel_types, vec![PixelType::FLOAT]);
        assert_eq!(
            load_multipart_part(&mut selective_headers, 2, &file)
                .unwrap_err()
                .code,
            ErrorCode::INVALID_ARGUMENT
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
