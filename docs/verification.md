# Verification

[Overview](../README.md) · [Guide](guide.md) · **Verification** · [Performance](performance.md)

---

tinyexr_nrp is checked against tinyexr itself: both are run over the same synthetic images
and files and their output compared byte for byte.

## The comparison

`compare/compare.sh` builds `compare/dump_tinyexr.cc` against `tinyexr.h` from a checkout of
tinyexr v3.4.0, without threads, with miniz for its ZIP compression and with
`TINYEXR_USE_ZFP` against zfp 1.0.1, with memory that starts as zeros and wrapping overflow,
so that a damaged file has one answer; and `examples/dump.rs` against tinyexr_nrp's
`_as_tinyexr` functions. It runs both and compares what they print:

```sh
MINIZ=<checkout of miniz> ZFP=<checkout of zfp 1.0.1> \
    compare/compare.sh <checkout of tinyexr> [count] [seeds] [dumps]
```

Both programs generate the same data from each seed and print a running hash of what they
produce:

- **tinyexr_write**: synthetic images saved as `.exr` files: scanlines and tiles with mip
  and rip levels; half, float and integer channels, kept or converted; each of the eight
  compressions; custom attributes; and images of one, three and four floats saved with
  `save_floats`. The bytes of each file are compared, or the message where tinyexr will not
  save an image.

- **tinyexr_read**: the files tinyexr saved, now and then with the data window moved or the
  lines in the other order, a third of them damaged (bits flipped, the end cut off, and
  values of the header, the offset table and the blocks randomised). For each, the version,
  the header, every channel or tile, and the image as RGBA floats are compared, or the code
  and message.

- **tinyexr_parts**: files of one to four parts saved, a third damaged, each read twice by
  tinyexr, with zeros and with ones past its end.

- **tinyexr_deep**: synthetic deep files of scanlines and of tiles, with one to three
  channels of every type, every compression and up to forty samples a pixel, a third
  damaged, read as for tinyexr_parts.

- **tinyexr_zfp_write** and **tinyexr_zfp_read**: the same as tinyexr_write and
  tinyexr_read for images made for ZFP compression, with settings at a fixed rate, a
  precision or an accuracy, among them zero, negative, infinite and NaN values, and now and
  then missing or malformed.

## Results

| Dump | Seeds | Input | Result |
|---|---|---|---|
| tinyexr_write | 3 | 48,000 images | 42,759 saved as identical files (1.24 GB), 5,241 refused with the same message; 10,543 more saved with `save_floats`, identically |
| tinyexr_read | 3 | 48,000 files (1.20 GB) | 38,299 headers read identically; of those, 30,142 images read identically and 8,132 refused with the same code and message; 25,674 read as RGBA identically |
| tinyexr_parts | 3 | 48,000 files (159 MB) | 31,225 saved as identical files and 16,775 refused with the same message; every file read identically, and the 139 readings that depended on memory past the end of the file or stopped tinyexr refused |
| tinyexr_deep | 3 | 48,000 files (141 MB) | Every file read identically; the 808 readings that depended on memory outside the file or stopped tinyexr refused |
| tinyexr_zfp_write | 3 | 48,000 images | 34,433 saved as identical files (178 MB), 13,567 refused with the same message; 10,447 more saved with `save_floats` |
| tinyexr_zfp_read | 3 | 48,000 files (153 MB) | 32,269 headers read identically; 23,653 images read identically and 8,588 refused with the same code and message |

A few files in each read dump, under 0.2%, declare sizes too large to decode in the
comparison and are skipped by both programs.

For version 0.1, the C++ was also built with clang and run under gcc's address sanitizer,
the larger runs were repeated with Rust's overflow checks enabled, the 379 sample,
regression and fuzzing files in tinyexr's own repository were read identically, and files
were exchanged with ffmpeg, whose `.exr` reader is its own, in both directions, bit for bit.

## Tests

`cargo test` needs no C++ compiler. `tests/golden.rs` checks tinyexr_nrp against hashes and
files recorded from tinyexr (`tests/data/`), reads badly damaged files to check that each is
read or refused without a panic, and the examples in the README and the Guide are run as
tests. `tests/data/README.md` explains how to record the data again.
