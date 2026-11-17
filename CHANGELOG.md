# Changelog

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
