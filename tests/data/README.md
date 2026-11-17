# Recorded results

These files are what `compare/dump_tinyexr.cc` prints when built against `tinyexr.h` v3.4.0,
and the files it wrote; `tests/golden.rs` checks that tinyexr_nrp prints the same:

- `tinyexr_write_expected.txt`: the `.exr` files tinyexr saves of synthetic images.

- `exr_files.bin` and `tinyexr_read_expected.txt`: such files, a third of them damaged, and
  what tinyexr reads of each.

- `exr_parts.bin` and `tinyexr_parts_expected.txt`: the same for files of several parts.

- `exr_deep.bin` and `tinyexr_deep_expected.txt`: the same for deep files, which the C
  program assembles itself, as tinyexr writes none.

- `tinyexr_zfp_write_expected.txt`, `exr_zfp.bin` and `tinyexr_zfp_read_expected.txt`: the
  same for images made for ZFP compression.

To record them again, run the comparison once, which builds the original's side, then run
that program directly:

    MINIZ=<checkout of miniz> ZFP=<checkout of zfp 1.0.1> compare/compare.sh <checkout of tinyexr> 3000 1 "tinyexr_write"
    ref="${TMPDIR:-/tmp}/tinyexr_nrp-compare/dump-tinyexr"
    "$ref" tinyexr_write_small 5 300 > tests/data/tinyexr_write_expected.txt
    "$ref" tinyexr_read_small 11 100 tests/data/exr_files.bin > tests/data/tinyexr_read_expected.txt
    "$ref" tinyexr_parts 7 60 tests/data/exr_parts.bin > tests/data/tinyexr_parts_expected.txt
    "$ref" tinyexr_deep 13 80 tests/data/exr_deep.bin > tests/data/tinyexr_deep_expected.txt
    "$ref" tinyexr_zfp_write 5 300 > tests/data/tinyexr_zfp_write_expected.txt
    "$ref" tinyexr_zfp_read 11 100 tests/data/exr_zfp.bin > tests/data/tinyexr_zfp_read_expected.txt

All test data is generated from seeds; no external files are used.
