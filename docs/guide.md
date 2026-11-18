# Guide

[Overview](../README.md) · **Guide** · [Verification](verification.md) · [Performance](performance.md)

---

- [RGBA floats](#rgba-floats)
- [Reading channels](#reading-channels)
- [Writing channels](#writing-channels)
- [Tiles and levels](#tiles-and-levels)
- [Several parts](#several-parts)
- [Deep images](#deep-images)
- [Compression settings](#compression-settings)
- [Codes and errors](#codes-and-errors)
- [Exactly as tinyexr](#exactly-as-tinyexr)
- [Correspondence with tinyexr](#correspondence-with-tinyexr)
- [Differences from tinyexr](#differences-from-tinyexr)

## RGBA floats

`load_rgba` reads a file as red, green, blue and alpha floats, four to a pixel, line by line
from the top. A file of one channel gives that channel four times; any other needs channels
named `R`, `G` and `B`, and its alpha is 1 where it has no `A`.

`save_floats` writes floats of one, three or four to a pixel (a single channel is named `A`,
as tinyexr names it), stored as halves or floats, compressed with ZIP, or uncompressed when
the image is smaller than sixteen pixels each way.

## Reading channels

For full control, a file is read in three steps: `parse_version` reads its first eight bytes,
`parse_header` its header, and `load_image` its pixels. Between the last two, a header's
`requested_pixel_types` may be changed from `PixelType::HALF` to `PixelType::FLOAT` for any
channel of halves, to read that channel as floats:

```rust
use tinyexr_nrp::{ChannelData, PixelType, load_image, parse_header, parse_version, save_floats};

let file = save_floats(&[0.5; 16 * 16 * 3], 16, 16, 3, true).unwrap();

let version = parse_version(&file).unwrap();
let mut header = parse_header(version, &file).unwrap();
assert_eq!(header.channels[0].name, "B"); // A file sorts its channels by name.
assert_eq!(header.channels[0].pixel_type, PixelType::HALF);

// Read the halves as floats.
header.requested_pixel_types.fill(PixelType::FLOAT);
let image = load_image(&mut header, &file).unwrap();
assert!(matches!(&image.images[0], ChannelData::Float(values) if values[0] == 0.5));
```

An `ExrImage` of scanlines holds one `ChannelData` for each channel, in the header's order,
`width` by `height` values. Every channel is taken to be sampled at every pixel.

## Writing channels

`save_image` writes an image with a header that gives each channel's name and stored type
(`ChannelInfo`), the type of the values handed over (`pixel_types`), the type to store them
as (`requested_pixel_types`), the compression and the data window:

```rust
use tinyexr_nrp::{
    Box2i, ChannelData, ChannelInfo, Compression, ExrHeader, ExrImage, PixelType, load_rgba,
    save_image,
};

let (width, height) = (32, 16);
let header = ExrHeader {
    channels: ["B", "G", "R"]
        .iter()
        .map(|name| ChannelInfo {
            name: name.to_string(),
            pixel_type: PixelType::HALF,
            x_sampling: 1,
            y_sampling: 1,
            p_linear: 0,
        })
        .collect(),
    pixel_types: vec![PixelType::FLOAT; 3],          // Floats are handed over,
    requested_pixel_types: vec![PixelType::HALF; 3], // and stored as halves.
    compression_type: Compression::PIZ,
    data_window: Box2i { min_x: 0, min_y: 0, max_x: width - 1, max_y: height - 1 },
    ..ExrHeader::default()
};
let image = ExrImage {
    images: vec![ChannelData::Float(vec![0.25; (width * height) as usize]); 3],
    width,
    height,
    num_channels: 3,
    ..ExrImage::default()
};
let file = save_image(&image, &header).unwrap();
assert_eq!(load_rgba(&file).unwrap().data[..4], [0.25, 0.25, 0.25, 1.0]);
```

Values handed over as floats may be stored as halves and the other way round; 32-bit
integers are stored as they are. An image whose channels hold too few values, or values of
another type than `pixel_types` says, is refused.

## Tiles and levels

A tiled image (`tiled` set, with `tile_size_x`, `tile_size_y` and a `TileLevelMode`)
holds its pixels in `ExrImage::tiles`, and its further mip or rip levels in `next_level`.
`tile_levels` gives the levels a header implies, in the order a file stores them and
`save_image` expects them, with each level's size and its number of tiles across and down.
Every tile holds `tile_size_x` by `tile_size_y` values a channel, of which its `width` by
`height` are inside the image; tiles at the right and bottom edges are smaller.

```rust
use tinyexr_nrp::{Box2i, ExrHeader, TileLevelMode, tile_levels};

let header = ExrHeader {
    data_window: Box2i { min_x: 0, min_y: 0, max_x: 99, max_y: 49 },
    tiled: true,
    tile_size_x: 32,
    tile_size_y: 32,
    tile_level_mode: TileLevelMode::MIPMAP_LEVELS,
    ..ExrHeader::default()
};
let levels = tile_levels(&header).unwrap();
assert_eq!((levels[0].width, levels[0].num_x_tiles, levels[0].num_y_tiles), (100, 4, 2));
assert_eq!(levels.len(), 7); // 100 by 50 down to 1 by 1.
```

## Several parts

`parse_multipart_headers` gives the header of each part of a file of several parts, and
`load_multipart_images` the pixels of each; `save_multipart_images` writes parts, each an
image with its header. Each part needs a `name`, and no two parts may share one.

## Deep images

`load_deep` reads a deep image, of scanlines or of tiles: for each channel and line, every
pixel's samples one after another, with `offset_table` giving, for each line, the number of
samples up to and including each pixel. tinyexr does not write deep images.

## Compression settings

The levels and settings of the lossy compressions are attributes of the header, in its
`custom_attributes`:

- **DWAA and DWAB**: `dwa_compression_level_attribute(level)` makes the
  `dwaCompressionLevel` attribute; without it the level is OpenEXR's default, 45. Higher
  levels make smaller files with more loss.

- **ZFP**: `ZfpSettings::attributes()` makes the attributes for a fixed rate, a precision or
  an accuracy (`ZfpSettings::Rate`, `Precision` or `Accuracy`); without them tinyexr writes
  at 2 bits a value. ZFP stores floats only, in blocks of 4 by 4 pixels.

## Codes and errors

The numbers a file stores for a channel's type, its compression and its tile modes are
newtypes with a constant for each value that has a meaning: `PixelType`, `Compression`,
`TileLevelMode` and `TileRoundingMode`. They hold any number a file stores, so a header
that is read keeps what the file says; `name()` gives the constant's name, if there is one.

An error is an `ExrError`: an `ErrorCode`, again with tinyexr's numbers (`ErrorCode::
INVALID_DATA` and the rest), and tinyexr's message. `parse_version` gives just the code, as
tinyexr does. The messages for what tinyexr_nrp refuses where tinyexr would not are
constants: `PAST_THE_END`, `ZFP_NOT_MULTIPLE_OF_FOUR`, `ZFP_TOO_LARGE` and
`DWA_INVALID_LEVEL`.

```rust
use tinyexr_nrp::{ErrorCode, load_rgba, parse_version};

assert_eq!(parse_version(b"not an .exr file"), Err(ErrorCode::INVALID_MAGIC_NUMBER));

let error = load_rgba(b"not an .exr file").unwrap_err();
assert_eq!(error.code, ErrorCode::INVALID_MAGIC_NUMBER);
assert_eq!(error.code.name(), Some("INVALID_MAGIC_NUMBER"));
```

## Exactly as tinyexr

tinyexr departs from the format in a few places (see
[Differences](#differences-from-tinyexr)). The functions whose names end in `_as_tinyexr`
(`load_rgba_as_tinyexr`, `parse_header_as_tinyexr`, `load_image_as_tinyexr`,
`save_image_as_tinyexr`, and those for several parts) follow tinyexr there too, and refuse
DWAA and DWAB as it does. The comparison with tinyexr runs these.

## Correspondence with tinyexr

| tinyexr | tinyexr_nrp |
|---|---|
| `LoadEXRFromMemory` | `load_rgba` |
| `SaveEXRToMemory` | `save_floats` |
| `ParseEXRVersionFromMemory`, `IsEXRFromMemory` | `parse_version`, `is_exr` |
| `ParseEXRHeaderFromMemory` | `parse_header` |
| `LoadEXRImageFromMemory` | `load_image` |
| `SaveEXRImageToMemory` | `save_image` |
| `ParseEXRMultipartHeaderFromMemory` | `parse_multipart_headers` |
| `LoadEXRMultipartImageFromMemory` | `load_multipart_images` |
| `SaveEXRMultipartImageToMemory` | `save_multipart_images` |
| `LoadDeepEXR` | `load_deep` |
| `EXRHeader`, `EXRImage`, `EXRTile`, `EXRChannelInfo`, `EXRAttribute`, `EXRVersion` | `ExrHeader`, `ExrImage`, `ExrTile`, `ChannelInfo`, `Attribute`, `ExrVersion` |
| `TINYEXR_PIXELTYPE_*`, `TINYEXR_COMPRESSIONTYPE_*`, `TINYEXR_TILE_*` | `PixelType`, `Compression`, `TileLevelMode`, `TileRoundingMode` |
| `TINYEXR_ERROR_*` and the `err` string | `ExrError` |

## Differences from tinyexr

The functions ending in `_as_tinyexr` do exactly as tinyexr does in the first two respects:

- **Blocks that do not come out smaller.** The format stores a block uncompressed when
  compressing it does not make it smaller, and a reader takes a block as long as its pixels
  to be uncompressed. tinyexr does both for RLE, ZIP and PIZ but neither for PXR24, B44 and
  B44A, so other readers misread some of its files and it misreads some of theirs. Here the
  format is followed.

- **DWAA and DWAB**, which tinyexr refuses, are read and written as OpenEXR does, at the
  level of the header's `dwaCompressionLevel`.

Whatever the function:

- **Damaged files that stop tinyexr**, or have it read past the end of the file, are
  refused. Its readers of files of several parts and of deep images read past the end of a
  damaged file without checking; tinyexr_nrp performs those reads with zeros and then with
  ones past the end, and gives the answer where both agree (it is then tinyexr's answer,
  whatever its memory holds); where they do not, the file is refused with `PAST_THE_END`.

- **What the writer is handed** is checked where tinyexr trusts it: a channel with too few
  values, or a type that is not the values' type, is refused.

- **ZFP that tinyexr cannot write** is refused with a message of tinyexr_nrp's own: a block
  or a tile not of whole blocks of four by four (`ZFP_NOT_MULTIPLE_OF_FOUR`; tinyexr writes
  a length it never set), and settings that need more than a gigabyte for a block
  (`ZFP_TOO_LARGE`).

- A small PIZ block of many different values, for which tinyexr writes past its buffer, is
  written as tinyexr would write it with room enough.
