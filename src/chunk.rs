//! The blocks of a file: where they are (the offset tables), and each of them into the
//! image. A port of `DecodeEXRImage`, `DecodeChunk`, `DecodeTiledLevel` and the functions
//! around them in tinyexr v3.4.0 (`tinyexr.h`).

use super::decode::{
    allocate_image, compute_channel_layout, decode_pixel_data, decode_tiled_pixel_data,
};
use super::zfp::{ZfpParam, find_zfp_compression_param};
use super::{
    Compression, ErrorCode, ExrError, ExrHeader, ExrImage, ExrTile, LineOrder, TileLevelMode,
    TileRoundingMode,
};

const EXR_VERSION_SIZE: usize = 8;

/// Larger images and tiles than this are refused (`TINYEXR_DIMENSION_THRESHOLD`).
const DIMENSION_THRESHOLD: i64 = 1024 * 8192;

/// `OffsetData`: where each block of a file starts: `offsets[level][tile y][tile x]`, or
/// `offsets[0][0][block]` for a file of scanlines.
#[derive(Default)]
pub(super) struct OffsetData {
    pub(super) offsets: Vec<Vec<Vec<u64>>>,
    pub(super) num_x_levels: i32,
    pub(super) num_y_levels: i32,
}

/// How many scanlines a block holds, by compression. (tinyexr has DWAA and DWAB as it has
/// any compression it does not know, of one line.)
pub(super) fn num_scanline_blocks(compression_type: Compression, as_tinyexr: bool) -> i32 {
    if !as_tinyexr && compression_type == Compression::DWAA {
        32
    } else if !as_tinyexr && compression_type == Compression::DWAB {
        256
    } else if [Compression::ZIP, Compression::ZFP, Compression::PXR24].contains(&compression_type) {
        16
    } else if [Compression::PIZ, Compression::B44, Compression::B44A].contains(&compression_type) {
        32
    } else {
        1
    }
}

/// `LevelIndex`.
pub(super) fn level_index(
    lx: i32,
    ly: i32,
    tile_level_mode: TileLevelMode,
    num_x_levels: i32,
) -> i32 {
    match tile_level_mode {
        TileLevelMode::ONE_LEVEL => 0,
        TileLevelMode::MIPMAP_LEVELS => lx,
        TileLevelMode::RIPMAP_LEVELS => lx.wrapping_add(ly.wrapping_mul(num_x_levels)),
        _ => -1,
    }
}

/// `LevelSize`: the size of a level of an image.
fn level_size(toplevel_size: i32, level: i32, tile_rounding_mode: TileRoundingMode) -> i32 {
    if level < 0 {
        return -1;
    }

    let b = 1u32.wrapping_shl(level as u32) as i32;
    // (The original divides by zero for a level of 32 or more, which no image has.)
    if b == 0 {
        return -1;
    }
    let mut level_size = toplevel_size.wrapping_div(b);

    if tile_rounding_mode == TileRoundingMode::ROUND_UP
        && level_size.wrapping_mul(b) < toplevel_size
    {
        level_size += 1;
    }

    level_size.max(1)
}

/// `DecodeTiledLevel`: the tiles of one level of a tiled image.
// (Its parameters are tinyexr's, each a separate part of the state; one pass, step for step as
// tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_tiled_level(
    exr_image: &mut ExrImage,
    exr_header: &ExrHeader,
    offset_data: &OffsetData,
    channel_offset_list: &[usize],
    pixel_data_size: i32,
    head: &[u8],
    zfp: Option<&ZfpParam>,
    as_tinyexr: bool,
) -> Result<(), ExrError> {
    const EF_INVALID_DATA: u32 = 1;
    const EF_INSUFFICIENT_DATA: u32 = 2;
    const EF_FAILED_TO_DECODE: u32 = 4;
    let size = head.len();
    let num_channels = exr_header.channels.len() as i32;

    let level_index = level_index(
        exr_image.level_x,
        exr_image.level_y,
        exr_header.tile_level_mode,
        offset_data.num_x_levels,
    );
    let Some(level_offsets) = offset_data.offsets.get(level_index as usize) else {
        return Err(ErrorCode::INVALID_DATA.into());
    };
    let num_y_tiles = level_offsets.len() as i32;
    if num_y_tiles < 1 {
        return Err(ErrorCode::INVALID_DATA.into());
    }
    let num_x_tiles = level_offsets[0].len() as i32;
    if num_x_tiles < 1 {
        return Err(ErrorCode::INVALID_DATA.into());
    }
    let num_tiles = num_x_tiles.wrapping_mul(num_y_tiles);

    let mut error_flag = 0u32;

    // Although the spec says: "...the data window is subdivided into an array of smaller
    // rectangles...", the IlmImf library allows the dimensions of the tile to be larger (or
    // equal) than the dimensions of the data window.

    // (The original gives every tile its memory before it looks at the tile, including the
    // tiles it then finds wrong; a tile here is given its memory when it is found right.
    // What comes back is the same: an image with a wrong tile is refused whole.)
    let types_are_valid = allocate_image(
        &exr_header.channels,
        &exr_header.requested_pixel_types,
        0,
        0,
    )
    .is_some();
    exr_image.tiles = Vec::new();

    for tile_idx in 0..num_tiles.max(0) {
        let mut tile = ExrTile::default();
        'tile: {
            if !types_are_valid {
                error_flag |= EF_INVALID_DATA;
                break 'tile;
            }

            // 16 byte: tile coordinates
            // 4 byte : data size
            // ~      : data(uncompressed or compressed)
            let x_tile = tile_idx % num_x_tiles;
            let y_tile = tile_idx / num_x_tiles;
            // Seek to offset value in file.
            let Some(&offset) = level_offsets
                .get(y_tile as usize)
                .and_then(|row| row.get(x_tile as usize))
            else {
                error_flag |= EF_INVALID_DATA;
                break 'tile;
            };
            if offset.wrapping_add(4 * 5) > size as u64 {
                // Insufficient data size.
                error_flag |= EF_INSUFFICIENT_DATA;
                break 'tile;
            }

            let data_size = size - (offset as usize + 4 * 5);
            let mut data_ptr = offset as usize;

            let int_at = |at: usize| {
                i32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]])
            };
            let tile_coordinates = [
                int_at(data_ptr),
                int_at(data_ptr + 4),
                int_at(data_ptr + 8),
                int_at(data_ptr + 12),
            ];

            if tile_coordinates[2] != exr_image.level_x {
                // Invalid data.
                error_flag |= EF_INVALID_DATA;
                break 'tile;
            }
            if tile_coordinates[3] != exr_image.level_y {
                // Invalid data.
                error_flag |= EF_INVALID_DATA;
                break 'tile;
            }

            let data_len = int_at(data_ptr + 16);

            if data_len < 2 || data_len as usize > data_size {
                // Insufficient data size.
                error_flag |= EF_INSUFFICIENT_DATA;
                break 'tile;
            }

            // Move to data addr: 20 = 16 + 4;
            data_ptr += 20;
            let Some(mut images) = allocate_image(
                &exr_header.channels,
                &exr_header.requested_pixel_types,
                exr_header.tile_size_x,
                exr_header.tile_size_y,
            ) else {
                error_flag |= EF_INVALID_DATA;
                break 'tile;
            };
            let size = decode_tiled_pixel_data(
                &mut images,
                &exr_header.requested_pixel_types,
                &head[data_ptr..],
                data_len as usize,
                exr_header.compression_type,
                exr_image.width,
                exr_image.height,
                tile_coordinates[0],
                tile_coordinates[1],
                exr_header.tile_size_x,
                exr_header.tile_size_y,
                pixel_data_size as usize,
                &exr_header.channels,
                channel_offset_list,
                zfp,
                as_tinyexr,
            );
            tile.images = images;

            if let Some((width, height)) = size {
                (tile.width, tile.height) = (width, height);
            } else {
                // Failed to decode tile data.
                error_flag |= EF_FAILED_TO_DECODE;
            }

            tile.offset_x = tile_coordinates[0];
            tile.offset_y = tile_coordinates[1];
            tile.level_x = tile_coordinates[2];
            tile.level_y = tile_coordinates[3];
        }
        exr_image.tiles.push(tile);
    }

    exr_image.num_channels = num_channels;

    if error_flag == 0 {
        return Ok(());
    }
    let mut message = String::new();
    if error_flag & EF_INSUFFICIENT_DATA != 0 {
        message.push_str("Insufficient data length.\n");
    }
    if error_flag & EF_FAILED_TO_DECODE != 0 {
        message.push_str("Failed to decode tile data.\n");
    }
    Err(ExrError::new(ErrorCode::INVALID_DATA, message))
}

/// `DecodeChunk`: every block of the file `head` into `exr_image`.
// (One pass, step for step as tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_lines)]
pub(super) fn decode_chunk(
    exr_header: &mut ExrHeader,
    offset_data: &OffsetData,
    head: &[u8],
    as_tinyexr: bool,
) -> Result<ExrImage, ExrError> {
    let mut exr_image = ExrImage::default();
    let size = head.len();
    let num_channels = exr_header.channels.len() as i32;

    let num_scanline_blocks = num_scanline_blocks(exr_header.compression_type, as_tinyexr);

    let zfp = if exr_header.compression_type == Compression::ZFP {
        let param = find_zfp_compression_param(&exr_header.custom_attributes)
            .map_err(|message| ExrError::new(ErrorCode::INVALID_HEADER, message))?;
        Some(param)
    } else {
        None
    };

    if exr_header.data_window.max_x < exr_header.data_window.min_x
        || exr_header.data_window.max_y < exr_header.data_window.min_y
    {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Invalid data window.\n",
        ));
    }

    let data_width =
        i64::from(exr_header.data_window.max_x) - i64::from(exr_header.data_window.min_x) + 1;
    let data_height =
        i64::from(exr_header.data_window.max_y) - i64::from(exr_header.data_window.min_y) + 1;

    if data_width <= 0 {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Invalid data window width.\n",
        ));
    }

    if data_height <= 0 {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Invalid data window height.\n",
        ));
    }

    // Do not allow too large data_width and data_height. header invalid?
    {
        if data_width > DIMENSION_THRESHOLD || data_height > DIMENSION_THRESHOLD {
            return Err(ExrError::new(
                ErrorCode::INVALID_DATA,
                format!(
                    "data_with or data_height too large. data_width: {data_width}, data_height = {data_height}\n"
                ),
            ));
        }
        if exr_header.tiled
            && (i64::from(exr_header.tile_size_x) > DIMENSION_THRESHOLD
                || i64::from(exr_header.tile_size_y) > DIMENSION_THRESHOLD)
        {
            return Err(ExrError::new(
                ErrorCode::INVALID_DATA,
                format!(
                    "tile with or tile height too large. tile width: {}, tile height = {}\n",
                    exr_header.tile_size_x, exr_header.tile_size_y
                ),
            ));
        }
    }

    // (For a tiled file whose table has no level, the original reads a table that is not
    // there; it is not used for a tiled file.)
    let no_offsets = Vec::new();
    let offsets = offset_data
        .offsets
        .first()
        .and_then(|level| level.first())
        .unwrap_or(&no_offsets);
    let num_blocks = offsets.len();

    let Some((channel_offset_list, pixel_data_size)) = compute_channel_layout(&exr_header.channels)
    else {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Failed to compute channel layout.\n",
        ));
    };

    let mut invalid_data = false;

    if exr_header.tiled {
        // value check
        if exr_header.tile_size_x < 0 {
            return Err(ExrError::new(
                ErrorCode::INVALID_HEADER,
                format!("Invalid tile size x : {}\n", exr_header.tile_size_x),
            ));
        }

        if exr_header.tile_size_y < 0 {
            return Err(ExrError::new(
                ErrorCode::INVALID_HEADER,
                format!("Invalid tile size y : {}\n", exr_header.tile_size_y),
            ));
        }
        let top_width = exr_header
            .data_window
            .max_x
            .wrapping_sub(exr_header.data_window.min_x)
            .wrapping_add(1);
        let top_height = exr_header
            .data_window
            .max_y
            .wrapping_sub(exr_header.data_window.min_y)
            .wrapping_add(1);
        // The levels, in the order of the file: (level x, level y) of each.
        let mut levels = Vec::new();
        if exr_header.tile_level_mode == TileLevelMode::RIPMAP_LEVELS {
            for level_y in 0..offset_data.num_y_levels {
                for level_x in 0..offset_data.num_x_levels {
                    levels.push((level_x, level_y));
                }
            }
        } else {
            for level in 0..offset_data.num_x_levels {
                levels.push((level, level));
            }
        }
        // (The original chains the levels as it goes; here they are chained at the end.)
        let mut decoded = Vec::with_capacity(levels.len());
        for (i, &(level_x, level_y)) in levels.iter().enumerate() {
            let mut level_image = if i == 0 {
                std::mem::take(&mut exr_image)
            } else {
                ExrImage::default()
            };
            level_image.width = level_size(top_width, level_x, exr_header.tile_rounding_mode);
            if level_image.width < 1 {
                return Err(ErrorCode::INVALID_DATA.into());
            }
            level_image.height = level_size(top_height, level_y, exr_header.tile_rounding_mode);
            if level_image.height < 1 {
                return Err(ErrorCode::INVALID_DATA.into());
            }
            level_image.level_x = level_x;
            level_image.level_y = level_y;

            decode_tiled_level(
                &mut level_image,
                exr_header,
                offset_data,
                &channel_offset_list,
                pixel_data_size,
                head,
                zfp.as_ref(),
                as_tinyexr,
            )?;
            decoded.push(level_image);
        }
        if let Some(first) = decoded.into_iter().rev().reduce(|next, mut level| {
            level.next_level = Some(Box::new(next));
            level
        }) {
            exr_image = first;
        }
    } else {
        // scanline format
        // Don't allow too large image(256GB * pixel_data_size or more). Workaround for #104.
        let total_data_len = (data_width as u64)
            .wrapping_mul(data_height as u64)
            .wrapping_mul(num_channels as u64);
        let total_data_len_overflown = total_data_len >= 0x0040_0000_0000;
        if total_data_len == 0 || total_data_len_overflown {
            return Err(ExrError::new(
                ErrorCode::INVALID_DATA,
                format!(
                    "Image data size is zero or too large: width = {data_width}, height = {data_height}, channels = {num_channels}\n"
                ),
            ));
        }

        let Some(images) = allocate_image(
            &exr_header.channels,
            &exr_header.requested_pixel_types,
            data_width as i32,
            data_height as i32,
        ) else {
            return Err(ExrError::new(
                ErrorCode::INVALID_DATA,
                format!(
                    "Failed to allocate memory for Images. Maybe EXR header is corrupted or Image data size is too large: width = {data_width}, height = {data_height}, channels = {num_channels}\n"
                ),
            ));
        };
        exr_image.images = images;

        for y in 0..num_blocks as i32 {
            let y_idx = y as usize;

            if offsets[y_idx].wrapping_add(4 * 2) > size as u64 {
                invalid_data = true;
            } else {
                // 4 byte: scan line
                // 4 byte: data size
                // ~     : pixel data(uncompressed or compressed)
                let data_size = size - (offsets[y_idx] as usize + 4 * 2);
                let mut data_ptr = offsets[y_idx] as usize;

                let int_at = |at: usize| {
                    i32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]])
                };
                let mut line_no = int_at(data_ptr);
                let data_len = int_at(data_ptr + 4);

                if data_len as isize as usize > data_size {
                    invalid_data = true;
                } else if !(-(2 << 20)..=(2 << 20)).contains(&line_no) {
                    // Too large value for line_no (2M+ lines).
                    invalid_data = true;
                } else if data_len == 0 {
                    // TODO(syoyo): May be ok to raise the threshold for example `data_len < 4`
                    invalid_data = true;
                } else {
                    // line_no may be negative.
                    let end_line_no = line_no
                        .wrapping_add(num_scanline_blocks)
                        .min(exr_header.data_window.max_y.wrapping_add(1));

                    let num_lines = end_line_no.wrapping_sub(line_no);

                    if num_lines <= 0 {
                        invalid_data = true;
                    } else {
                        // Move to data addr: 8 = 4 + 4;
                        data_ptr += 8;

                        // Adjust line_no with data_window.bmin.y

                        // overflow check
                        let lno = i64::from(line_no) - i64::from(exr_header.data_window.min_y);
                        line_no = if lno.abs() > i64::from(i32::MAX) {
                            -1
                        } else {
                            lno as i32
                        };

                        if line_no < 0
                            || !decode_pixel_data(
                                &mut exr_image.images,
                                &exr_header.requested_pixel_types,
                                &head[data_ptr..],
                                data_len as usize,
                                exr_header.compression_type,
                                LineOrder::INCREASING_Y,
                                data_width as i32,
                                data_height as i32,
                                data_width as i32,
                                y,
                                line_no,
                                num_lines,
                                pixel_data_size as usize,
                                &exr_header.channels,
                                &channel_offset_list,
                                zfp.as_ref(),
                                as_tinyexr,
                            )
                        {
                            invalid_data = true;
                        }
                    }
                }
            }
        }
    }

    if invalid_data {
        return Err(ExrError::new(
            ErrorCode::INVALID_DATA,
            "Invalid/Corrupted data found when decoding pixels.\n",
        ));
    }

    // Overwrite `pixel_type` with `requested_pixel_type`.
    exr_header
        .pixel_types
        .clone_from(&exr_header.requested_pixel_types);

    {
        exr_image.num_channels = num_channels;

        exr_image.width = data_width as i32;
        exr_image.height = data_height as i32;
    }

    Ok(exr_image)
}

/// `ReconstructLineOffsets`: the table of a file of scanlines whose own is not filled in,
/// by walking the blocks from `marker`.
fn reconstruct_line_offsets(offsets: &mut [u64], head: &[u8], mut marker: usize) -> bool {
    let size = head.len();
    if marker == 0 {
        return false;
    }

    for offset_slot in offsets.iter_mut() {
        let offset = marker;
        // Offset should not exceed whole EXR file/data size.
        if offset.checked_add(8).is_none_or(|end| end >= size) {
            return false;
        }

        let data_len = u32::from_le_bytes([
            head[marker + 4],
            head[marker + 5],
            head[marker + 6],
            head[marker + 7],
        ]) as usize;

        if data_len >= size {
            return false;
        }

        *offset_slot = offset as u64;

        marker += data_len + 8; // 8 = 4 bytes(y) + 4 bytes(data_len)
    }

    true
}

fn floor_log2(mut x: u32) -> i32 {
    // For x > 0, floorLog2(y) returns floor(log(x)/log(2)).
    let mut y = 0;
    while x > 1 {
        y += 1;
        x >>= 1;
    }
    y
}

fn ceil_log2(mut x: u32) -> i32 {
    // For x > 0, ceilLog2(y) returns ceil(log(x)/log(2)).
    let mut y = 0;
    let mut r = 0;
    while x > 1 {
        if x & 1 != 0 {
            r = 1;
        }

        y += 1;
        x >>= 1;
    }
    y + r
}

fn round_log2(x: i32, tile_rounding_mode: TileRoundingMode) -> i32 {
    if tile_rounding_mode == TileRoundingMode::ROUND_DOWN {
        floor_log2(x as u32)
    } else {
        ceil_log2(x as u32)
    }
}

/// `CalculateNumXLevels`.
fn calculate_num_x_levels(exr_header: &ExrHeader) -> i32 {
    let w = exr_header
        .data_window
        .max_x
        .wrapping_sub(exr_header.data_window.min_x)
        .wrapping_add(1);
    let h = exr_header
        .data_window
        .max_y
        .wrapping_sub(exr_header.data_window.min_y)
        .wrapping_add(1);

    match exr_header.tile_level_mode {
        TileLevelMode::ONE_LEVEL => 1,
        TileLevelMode::MIPMAP_LEVELS => round_log2(w.max(h), exr_header.tile_rounding_mode) + 1,
        TileLevelMode::RIPMAP_LEVELS => round_log2(w, exr_header.tile_rounding_mode) + 1,
        _ => -1,
    }
}

/// `CalculateNumYLevels`.
fn calculate_num_y_levels(exr_header: &ExrHeader) -> i32 {
    let w = exr_header
        .data_window
        .max_x
        .wrapping_sub(exr_header.data_window.min_x)
        .wrapping_add(1);
    let h = exr_header
        .data_window
        .max_y
        .wrapping_sub(exr_header.data_window.min_y)
        .wrapping_add(1);

    match exr_header.tile_level_mode {
        TileLevelMode::ONE_LEVEL => 1,
        TileLevelMode::MIPMAP_LEVELS => round_log2(w.max(h), exr_header.tile_rounding_mode) + 1,
        TileLevelMode::RIPMAP_LEVELS => round_log2(h, exr_header.tile_rounding_mode) + 1,
        _ => -1,
    }
}

/// `CalculateNumTiles`: how many tiles each level has along one side.
fn calculate_num_tiles(
    num_tiles: &mut [i32],
    toplevel_size: i32,
    size: i32,
    tile_rounding_mode: TileRoundingMode,
) -> bool {
    for (i, slot) in num_tiles.iter_mut().enumerate() {
        let l = level_size(toplevel_size, i as i32, tile_rounding_mode);
        if l < 0 {
            return false;
        }
        if size <= 0 {
            // (A tile size of 0 has the original divide by it.)
            return false;
        }
        if l > i32::MAX - size + 1 {
            return false;
        }

        *slot = (l + size - 1) / size;
    }
    true
}

/// `PrecalculateTileInfo`.
pub(super) fn precalculate_tile_info(exr_header: &ExrHeader) -> Option<(Vec<i32>, Vec<i32>)> {
    let num_x_levels = calculate_num_x_levels(exr_header);
    if num_x_levels < 0 {
        return None;
    }

    let num_y_levels = calculate_num_y_levels(exr_header);
    if num_y_levels < 0 {
        return None;
    }

    let mut num_x_tiles = vec![0; num_x_levels as usize];
    let mut num_y_tiles = vec![0; num_y_levels as usize];

    let w = exr_header
        .data_window
        .max_x
        .wrapping_sub(exr_header.data_window.min_x)
        .wrapping_add(1);
    let h = exr_header
        .data_window
        .max_y
        .wrapping_sub(exr_header.data_window.min_y)
        .wrapping_add(1);
    if !calculate_num_tiles(
        &mut num_x_tiles,
        w,
        exr_header.tile_size_x,
        exr_header.tile_rounding_mode,
    ) {
        return None;
    }

    if !calculate_num_tiles(
        &mut num_y_tiles,
        h,
        exr_header.tile_size_y,
        exr_header.tile_rounding_mode,
    ) {
        return None;
    }

    Some((num_x_tiles, num_y_tiles))
}

/// The shape of the offset table of a tiled file (`InitTileOffsets`): for each level, how
/// many rows of tiles it has and how many tiles a row. The second answer is the number of
/// tiles in all, as the original counts it.
pub(super) fn tile_offsets_shape(
    offset_data: &mut OffsetData,
    exr_header: &ExrHeader,
    num_x_tiles: &[i32],
    num_y_tiles: &[i32],
) -> (Vec<(usize, usize)>, i32) {
    let mut num_tile_blocks = 0i32;
    let mut shape = Vec::new();
    offset_data.num_x_levels = num_x_tiles.len() as i32;
    offset_data.num_y_levels = num_y_tiles.len() as i32;
    match exr_header.tile_level_mode {
        TileLevelMode::ONE_LEVEL | TileLevelMode::MIPMAP_LEVELS => {
            if offset_data.num_x_levels != offset_data.num_y_levels {
                return (shape, 0);
            }
            for l in 0..offset_data.num_x_levels as usize {
                shape.push((num_y_tiles[l] as usize, num_x_tiles[l] as usize));
                num_tile_blocks =
                    num_tile_blocks.wrapping_add(num_x_tiles[l].wrapping_mul(num_y_tiles[l]));
            }
        }
        TileLevelMode::RIPMAP_LEVELS => {
            let num_y = &num_y_tiles[..offset_data.num_y_levels as usize];
            let num_x = &num_x_tiles[..offset_data.num_x_levels as usize];
            for &tiles_y in num_y {
                for &tiles_x in num_x {
                    shape.push((tiles_y as usize, tiles_x as usize));
                    num_tile_blocks = num_tile_blocks.wrapping_add(tiles_x.wrapping_mul(tiles_y));
                }
            }
        }
        _ => return (shape, 0),
    }
    (shape, num_tile_blocks)
}

/// A level of a tiled image: which it is, its size in pixels, and how many tiles it has
/// across and down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileLevel {
    /// The level across.
    pub level_x: i32,
    /// The level down.
    pub level_y: i32,
    /// Its width in pixels.
    pub width: i32,
    /// Its height in pixels.
    pub height: i32,
    /// How many tiles it has across.
    pub num_x_tiles: i32,
    /// How many tiles it has down.
    pub num_y_tiles: i32,
}

/// The levels of a tiled image with this header, in the order a file has them, which is
/// the order `save_image` wants them in: one level, the mip levels from the
/// largest down, or the rip levels row after row. It goes by the header's data window, tile
/// size, level mode and rounding mode, as the original counts them (`PrecalculateTileInfo`
/// and `LevelSize`). `None` for sizes and modes no image has.
///
/// The tile at (x, y) of a level is `min(tile_size_x, width - x * tile_size_x)` pixels wide
/// and as much high, and its values are `tile_size_x` apart from line to line.
#[must_use]
pub fn tile_levels(exr_header: &ExrHeader) -> Option<Vec<TileLevel>> {
    let (num_x_tiles, num_y_tiles) = precalculate_tile_info(exr_header)?;
    let width = exr_header
        .data_window
        .max_x
        .wrapping_sub(exr_header.data_window.min_x)
        .wrapping_add(1);
    let height = exr_header
        .data_window
        .max_y
        .wrapping_sub(exr_header.data_window.min_y)
        .wrapping_add(1);
    let level = |lx: usize, ly: usize| TileLevel {
        level_x: lx as i32,
        level_y: ly as i32,
        width: level_size(width, lx as i32, exr_header.tile_rounding_mode),
        height: level_size(height, ly as i32, exr_header.tile_rounding_mode),
        num_x_tiles: num_x_tiles[lx],
        num_y_tiles: num_y_tiles[ly],
    };
    match exr_header.tile_level_mode {
        TileLevelMode::ONE_LEVEL | TileLevelMode::MIPMAP_LEVELS
            if num_x_tiles.len() == num_y_tiles.len() =>
        {
            Some((0..num_x_tiles.len()).map(|l| level(l, l)).collect())
        }
        TileLevelMode::RIPMAP_LEVELS => Some(
            (0..num_y_tiles.len())
                .flat_map(|ly| (0..num_x_tiles.len()).map(move |lx| (lx, ly)))
                .map(|(lx, ly)| level(lx, ly))
                .collect(),
        ),
        _ => None,
    }
}

/// `IsAnyOffsetsAreInvalid`.
fn is_any_offsets_are_invalid(offset_data: &OffsetData) -> bool {
    offset_data
        .offsets
        .iter()
        .flatten()
        .flatten()
        .any(|&offset| offset as i64 <= 0)
}

/// `isValidTile`.
fn is_valid_tile(
    exr_header: &ExrHeader,
    offset_data: &OffsetData,
    dx: i32,
    dy: i32,
    lx: i32,
    ly: i32,
) -> bool {
    if lx < 0 || ly < 0 || dx < 0 || dy < 0 {
        return false;
    }
    let num_x_levels = offset_data.num_x_levels;
    let num_y_levels = offset_data.num_y_levels;
    let has = |level: usize| {
        offset_data
            .offsets
            .get(level)
            .and_then(|rows| rows.get(dy as usize))
            .is_some_and(|row| row.len() > dx as usize)
    };
    match exr_header.tile_level_mode {
        TileLevelMode::ONE_LEVEL => lx == 0 && ly == 0 && has(0),
        TileLevelMode::MIPMAP_LEVELS => lx < num_x_levels && ly < num_y_levels && has(lx as usize),
        TileLevelMode::RIPMAP_LEVELS => {
            let idx = lx as usize + ly as usize * num_x_levels as usize;
            lx < num_x_levels && ly < num_y_levels && has(idx)
        }
        _ => false,
    }
}

/// `ReconstructTileOffsets`: the table of a tiled file whose own is not filled in, by
/// walking the tiles from `marker`.
fn reconstruct_tile_offsets(
    offset_data: &mut OffsetData,
    exr_header: &ExrHeader,
    head: &[u8],
    mut marker: usize,
    is_multi_part_file: bool,
    is_deep: bool,
) -> bool {
    let size = head.len();
    let num_x_levels = offset_data.num_x_levels;
    let count: usize = offset_data
        .offsets
        .iter()
        .flatten()
        .map(std::vec::Vec::len)
        .sum();
    for _ in 0..count {
        let tile_offset = marker as u64;

        if is_multi_part_file {
            if marker + 4 >= size {
                return false;
            }

            // int partNumber;
            marker += 4;
        }

        if marker + 4 * 4 >= size {
            return false;
        }

        let int_at =
            |at: usize| i32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
        let tile_x = int_at(marker);
        marker += 4;

        let tile_y = int_at(marker);
        marker += 4;

        let level_x = int_at(marker);
        marker += 4;

        let level_y = int_at(marker);
        marker += 4;

        if is_deep {
            if marker + 2 * 8 >= size {
                return false;
            }
            let long_at = |at: usize| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&head[at..at + 8]);
                i64::from_le_bytes(bytes)
            };
            let packed_offset_table_size = long_at(marker);
            marker += 8;

            let packed_sample_size = long_at(marker);
            marker += 8;

            // next Int64 is unpacked sample size - skip that too
            // (Sizes that take the place before the file, where the original then reads,
            // are refused.)
            let next = (marker as i64)
                .wrapping_add(packed_offset_table_size)
                .wrapping_add(packed_sample_size)
                .wrapping_add(8);
            if next < 0 || next as u64 >= size as u64 {
                return false;
            }
            marker = next as usize;
        } else {
            if marker + 4 >= size {
                return false;
            }

            let data_size = int_at(marker) as u32;
            marker += 4;

            marker += data_size as usize;

            if marker >= size {
                return false;
            }
        }

        if !is_valid_tile(exr_header, offset_data, tile_x, tile_y, level_x, level_y) {
            return false;
        }

        let level_idx = level_index(level_x, level_y, exr_header.tile_level_mode, num_x_levels);
        if level_idx < 0 {
            return false;
        }

        let Some(slot) = offset_data
            .offsets
            .get_mut(level_idx as usize)
            .and_then(|rows| rows.get_mut(tile_y as usize))
            .and_then(|row| row.get_mut(tile_x as usize))
        else {
            return false;
        };

        *slot = tile_offset;
    }
    true
}

/// One entry of an offset table, at `*marker` (`ReadOffsets`, and the same lines of
/// `DecodeEXRImage` for a file of scanlines).
fn read_offset(head: &[u8], marker: &mut usize) -> Result<u64, ExrError> {
    let size = head.len();
    let fail = |message: &str| Err(ExrError::new(ErrorCode::INVALID_DATA, message));
    let Some(&bytes) = head
        .get(*marker..)
        .filter(|rest| rest.len() > 8)
        .and_then(<[u8]>::first_chunk::<8>)
    else {
        return fail("Insufficient data size in offset table.");
    };
    let offset = u64::from_le_bytes(bytes);
    if offset >= size as u64 {
        return fail("Invalid offset value in DecodeEXRImage.");
    }
    *marker += 8;
    Ok(offset)
}

/// `DecodeEXRImage`: the pixels of the file `head`, whose offset table starts at `marker`.
// (One pass, step for step as tinyexr's, its state shared throughout.)
#[allow(clippy::too_many_lines)]
pub(crate) fn decode_exr_image(
    exr_header: &mut ExrHeader,
    head: &[u8],
    mut marker: usize,
    as_tinyexr: bool,
) -> Result<ExrImage, ExrError> {
    let size = head.len();
    let fail = |message: &str| Err(ExrError::new(ErrorCode::INVALID_DATA, message));
    if size <= EXR_VERSION_SIZE {
        return Err(ExrError::new(
            ErrorCode::INVALID_ARGUMENT,
            "Invalid argument for DecodeEXRImage().",
        ));
    }

    let num_scanline_blocks = num_scanline_blocks(exr_header.compression_type, as_tinyexr);

    if exr_header.data_window.max_x < exr_header.data_window.min_x
        || exr_header
            .data_window
            .max_x
            .wrapping_sub(exr_header.data_window.min_x)
            == i32::MAX
    {
        // Issue 63
        return fail("Invalid data width value");
    }
    let data_width =
        i64::from(exr_header.data_window.max_x) - i64::from(exr_header.data_window.min_x) + 1;
    if data_width <= 0 {
        return fail("Invalid data window width value");
    }

    if exr_header.data_window.max_y < exr_header.data_window.min_y
        || exr_header
            .data_window
            .max_y
            .wrapping_sub(exr_header.data_window.min_y)
            == i32::MAX
    {
        return fail("Invalid data height value");
    }

    let data_height =
        i64::from(exr_header.data_window.max_y) - i64::from(exr_header.data_window.min_y) + 1;

    if data_height <= 0 {
        return fail("Invalid data window height value");
    }

    // Do not allow too large data_width and data_height. header invalid?
    {
        if data_width > DIMENSION_THRESHOLD {
            return fail("data width too large.");
        }
        if data_height > DIMENSION_THRESHOLD {
            return fail("data height too large.");
        }
    }

    if exr_header.tiled {
        if i64::from(exr_header.tile_size_x) > DIMENSION_THRESHOLD {
            return fail("tile width too large.");
        }
        if i64::from(exr_header.tile_size_y) > DIMENSION_THRESHOLD {
            return fail("tile height too large.");
        }
    }

    // Read offset tables.
    let mut offset_data = OffsetData::default();
    if exr_header.tiled {
        let Some((num_x_tiles, num_y_tiles)) = precalculate_tile_info(exr_header) else {
            return fail("Failed to precalculate tile info.");
        };
        let (shape, num_blocks) =
            tile_offsets_shape(&mut offset_data, exr_header, &num_x_tiles, &num_y_tiles);
        if exr_header.chunk_count > 0 && exr_header.chunk_count != num_blocks {
            return fail("Invalid offset table size.");
        }

        // `ReadOffsets`. (The original makes the whole table first; here a table larger
        // than the file can hold stops where the file does.)
        for &(rows, columns) in &shape {
            let mut level = Vec::new();
            for _dy in 0..rows {
                let mut row = Vec::new();
                for _dx in 0..columns {
                    row.push(read_offset(head, &mut marker)?);
                }
                level.push(row);
            }
            offset_data.offsets.push(level);
        }
        if is_any_offsets_are_invalid(&offset_data)
            && !reconstruct_tile_offsets(
                &mut offset_data,
                exr_header,
                head,
                marker,
                exr_header.multipart,
                exr_header.non_image,
            )
        {
            return fail("Invalid Tile Offsets data.");
        }
    } else {
        let num_blocks = if exr_header.chunk_count > 0 {
            // Use `chunkCount` attribute.
            exr_header.chunk_count as usize
        } else {
            let mut num_blocks = data_height as usize / num_scanline_blocks as usize;
            if num_blocks * (num_scanline_blocks as usize) < data_height as usize {
                num_blocks += 1;
            }
            num_blocks
        };

        // `InitSingleResolutionOffsets`.
        offset_data.num_x_levels = 1;
        offset_data.num_y_levels = 1;

        let mut offsets = Vec::new();
        for _y in 0..num_blocks {
            offsets.push(read_offset(head, &mut marker)?);
        }

        // If line offsets are invalid, we try to reconstruct it.
        // See OpenEXR/IlmImf/ImfScanLineInputFile.cpp::readLineOffsets() for details.
        if offsets.iter().any(|&offset| offset as i64 <= 0) {
            // TODO(syoyo) Report as warning?
            if !reconstruct_line_offsets(&mut offsets, head, marker) {
                return fail("Cannot reconstruct lineOffset table in DecodeEXRImage.");
            }
        }
        offset_data.offsets = vec![vec![offsets]];
    }

    decode_chunk(exr_header, &offset_data, head, as_tinyexr)
}
