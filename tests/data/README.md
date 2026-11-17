# Recorded results

`tinyexr_write_expected.txt` is what `compare/dump_tinyexr.cc` prints of the `.exr` files
tinyexr saves of made-up images. `exr_files.bin` holds such files, a third of them damaged, as
the same program wrote them, and `tinyexr_read_expected.txt` what tinyexr reads of each, as it
prints it. `exr_parts.bin` with `tinyexr_parts_expected.txt`, and `exr_deep.bin` with
`tinyexr_deep_expected.txt`, are the same for files of many parts and for deep files (made up
by that program: tinyexr writes none). `tinyexr_zfp_write_expected.txt`, and `exr_zfp.bin`
with `tinyexr_zfp_read_expected.txt`, are the same for images made for ZFP compression.
`tests/golden.rs` checks that the port prints the same for all of them. To make them again,
run the comparison once (it builds the original's side), then that by itself:

    MINIZ=<checkout of miniz> ZFP=<checkout of zfp 1.0.1> compare/compare.sh <checkout of tinyexr> 3000 1 "tinyexr_write"
    ref="${TMPDIR:-/tmp}/tinyexr_nrp-compare/dump-tinyexr"
    "$ref" tinyexr_write_small 5 300 > tests/data/tinyexr_write_expected.txt
    "$ref" tinyexr_read_small 11 100 tests/data/exr_files.bin > tests/data/tinyexr_read_expected.txt
    "$ref" tinyexr_parts 7 60 tests/data/exr_parts.bin > tests/data/tinyexr_parts_expected.txt
    "$ref" tinyexr_deep 13 80 tests/data/exr_deep.bin > tests/data/tinyexr_deep_expected.txt
    "$ref" tinyexr_zfp_write 5 300 > tests/data/tinyexr_zfp_write_expected.txt
    "$ref" tinyexr_zfp_read 11 100 tests/data/exr_zfp.bin > tests/data/tinyexr_zfp_read_expected.txt

No picture or file from anywhere is in this repository: every file and stream the tests and
the comparison use is made up from a seed.
