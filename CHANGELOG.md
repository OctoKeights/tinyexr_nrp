# Changelog

## 0.3.0

- **Breaking: flags and codes as types.** `ExrHeader`'s `tiled`, `long_name`, `non_image` and
  `multipart` are `bool`s in place of `i32`s, and `line_order` is a `LineOrder` (with
  `INCREASING_Y`, `DECREASING_Y` and `RANDOM_Y`), which keeps any value a file holds, as the
  other codes do. `ChannelInfo::p_linear` stays a `u8`, since tinyexr keeps the byte the file
  holds. `parse_header`, `parse_header_as_tinyexr`, `parse_multipart_headers` and
  `parse_multipart_headers_as_tinyexr` take the `ExrVersion` by value (it is `Copy`).
  `ExrError` gains `From<ErrorCode>`, for an error with a code and no message.

- **Stricter lints, enforced in CI.** `Cargo.toml` turns on clippy's pedantic group for every
  target, and forbids `unsafe` code in the tests, examples and benchmarks as well as the
  library. Functions that return a `Result` document their errors.

- **Idiomatic internals.** The reader and writer pass their errors up as
  `Result<_, ExrError>` in place of a message string filled in along the way, with every
  message as tinyexr gives it; out-parameters for sizes and counts are return values.

- Output is unchanged, checked against tinyexr 3.4.0 on 288,000 synthetic files, and the
  benchmark runs 0.08% fewer instructions.

## 0.2.0

- **Breaking: an idiomatic API.** The functions drop tinyexr's `_from_memory` and
  `_to_memory` suffixes: `load_rgba`, `save_floats`, `parse_version`, `is_exr`,
  `parse_header`, `load_image`, `save_image`, `parse_multipart_headers`,
  `load_multipart_images`, `save_multipart_images` and `load_deep`, each with its
  `_as_tinyexr` twin where it had one. `load_rgba` returns an `RgbaImage` in place of a
  tuple. The numbers a file stores are newtypes with a constant for each meaningful value,
  `PixelType`, `Compression`, `TileLevelMode` and `TileRoundingMode`, in place of `i32`
  fields and `PIXELTYPE_*`, `COMPRESSIONTYPE_*` and `TILE_*` constants; errors carry an
  `ErrorCode` in place of an `i32` and the `ERROR_*` constants. `float_to_half_full` is
  `float_to_half`, and the `ZFP_COMPRESSIONTYPE_*` constants are private (`ZfpSettings`
  covers them).

- **Faster.** Uncompressed files save and load in about half the instructions they took,
  channels are copied a line at a time for every compression, and the 0.2 releases of
  miniz_nrp and openexr_nrp speed up ZIP, ZIPS, PXR24, DWAA and DWAB (see
  [Performance](docs/performance.md)). Output is unchanged, checked against tinyexr.

- **Documentation.** Every public item is documented, the README is a short overview, and
  the detail is in [Guide](docs/guide.md), [Verification](docs/verification.md) and
  [Performance](docs/performance.md).

## 0.1.0

- First release: the port of tinyexr's reader and writer, with DWAA and DWAB from openexr_nrp and ZFP from zfp_nrp, moved out of
  [rustbcx](https://github.com/OctoKeights/rustbcx), where it was `rustbcx::tinyexr`.
