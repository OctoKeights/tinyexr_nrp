# tinyexr_nrp

**OpenEXR files read and written in pure, safe Rust: a native Rust port of tinyexr.**

tinyexr_nrp is a line-for-line port of Syoyo Fujita's
[tinyexr](https://github.com/syoyo/tinyexr) v3.4.0 (`tinyexr.h`): it writes the `.exr` files
tinyexr writes, byte for byte, reads the values tinyexr reads, and refuses the files tinyexr
refuses with the same code and message. It also reads and writes DWAA and DWAB, which
tinyexr does not have, as OpenEXR does.

## Highlights

- **Every compression.** None, RLE, ZIPS, ZIP, PIZ, PXR24, B44 and B44A as tinyexr has them;
  DWAA and DWAB as OpenEXR 3.5 has them on x86-64
  ([openexr_nrp](https://github.com/OctoKeights/openexr_nrp)); and the ZFP compression tinyexr
  adds with `TINYEXR_USE_ZFP` ([zfp_nrp](https://github.com/OctoKeights/zfp_nrp)).

- **Every layout.** Scanlines and tiles with mip and rip levels, files of several parts, and
  deep images read; halves, floats and 32-bit integers.

- **Damaged files refused safely.** Where tinyexr would read past the end of a damaged file
  or stop, the port refuses it.

- **Dependencies of its own only:** [miniz_nrp](https://github.com/OctoKeights/miniz_nrp) for
  ZIP, openexr_nrp and zfp_nrp. No C compiler, no build script.

- **No `unsafe` code.** Enforced by `#![forbid(unsafe_code)]`.

## Installation

```toml
[dependencies]
tinyexr_nrp = "0.1"
```

## Quick start

```rust
use tinyexr_nrp::{load_exr_from_memory, save_exr_to_memory};

// An image of 8 by 4 pixels of RGBA floats.
let (width, height) = (8, 4);
let rgba: Vec<f32> = (0..width * height * 4).map(|i| i as f32 / 16.0).collect();

// Saved as floats (true for halves), as tinyexr's SaveEXRToMemory writes it.
let file = save_exr_to_memory(&rgba, width, height, 4, false).unwrap();
assert_eq!(&file[..4], &[0x76, 0x2f, 0x31, 0x01]);

// Read back as RGBA floats.
let (back, w, h) = load_exr_from_memory(&file).unwrap();
assert_eq!((w, h), (width, height));
assert_eq!(back, rgba);
```

`parse_exr_version_from_memory`, `parse_exr_header_from_memory` and
`load_exr_image_from_memory` give the header and then every channel as it is stored, or every
tile of every level; `save_exr_image_to_memory` writes an image the same way. The functions for
files of several parts and for deep images are beside them.

## What is ported

`tinyexr.h` as it builds without threads and with miniz for its ZIP compression: its readers
and writers of one part and of several, of scanlines and of tiles; its reader of deep images;
every compression it has, and its ZFP compression with `TINYEXR_USE_ZFP`.

## Differences from the original

The code is translated from C++ to Rust, and the port writes and reads what the original
does, with these exceptions, each of which a function ending in `_as_tinyexr` leaves out (and
does exactly as tinyexr does):

- **Blocks that do not come out smaller.** The format has a block stored as its pixels when
  compressing it does not make it smaller, and a reader take a block as long as its pixels for
  one. tinyexr does both for RLE, ZIP and PIZ and neither for PXR24, B44 and B44A, so that
  other readers misread some of its files and it misreads some of theirs. Here the format is
  followed.
- **DWAA and DWAB**, which tinyexr refuses, are read and written as OpenEXR does, at the
  level of loss of the header's `dwaCompressionLevel` (45 without it).

And whatever the function:

- **Damaged files that stop the original**, or have it read past the end of the file, are
  refused. Its readers of files of several parts and of deep images read past the end of a
  damaged file without a look; the port makes those reads with zeros and then with ones past
  the end, and where the answer is the same, it is given (it is the original's answer whatever
  its memory holds); where it is not, the file is refused with `PAST_THE_END`.
- **What the writer is handed** is looked at where the original trusts it: a channel with too
  few values, or a type that is not the values' type, is refused.
- **ZFP that tinyexr cannot write** is refused with a message of the port's own: a block or a
  tile not of whole blocks of four by four (`ZFP_NOT_MULTIPLE_OF_FOUR`; tinyexr writes a
  length it never set), and settings that ask for more than a gigabyte for a block
  (`ZFP_TOO_LARGE`).
- A small PIZ block of many different values, which the original writes past its buffer for,
  is written as the original would write it with room enough.

## Verification

`compare/compare.sh` builds `compare/dump_tinyexr.cc` against `tinyexr.h` from a checkout,
with miniz and zfp from theirs, and `examples/dump.rs` against the port; it runs both over the
same made-up images and files and compares what they print byte for byte:

```sh
MINIZ=<checkout of miniz> ZFP=<checkout of zfp 1.0.1> \
    compare/compare.sh <checkout of tinyexr> [count] [seeds] [dumps]
```

| Dump | What it covers | Last full run |
|---|---|---|
| `tinyexr_write` | Made-up images saved as `.exr`: scanlines and tiles with mip and rip levels; half, float and integer channels, kept or converted; each of the eight compressions; custom attributes; and images of one, three and four floats saved the short way. The bytes of each file are compared, or the message where tinyexr will not save an image | 48,000 images: 42,700 saved as identical files (1.3 gigabytes) and 5,300 refused with the same message; 10,600 more saved the short way. Every file read back by the port, and what is stored without loss read unchanged |
| `tinyexr_read` | The files tinyexr saved, now and then with the data window moved or the lines in the other order; a third damaged first (bits flipped, the end cut off, and values of the header, the table of offsets and the blocks made up). For each, the version, the header, every channel or tile, and the image as RGBA floats are compared, or the code and message | 48,000 files (1.2 gigabytes), of which 38,200 had a header tinyexr takes: 30,100 images read and 8,100 refused, each with the same code and message. And the 379 sample, regression and fuzzing files in tinyexr's own repository read identically |
| `tinyexr_parts` | Files of one to four parts saved, a third damaged, each read by tinyexr twice, with zeros and with ones past its end | 60,000 files: 39,000 saved as identical files and 21,000 refused with the same message; every file read identically, and the 169 readings that depended on memory past the end or stopped tinyexr refused by the port |
| `tinyexr_deep` | Made-up deep files of scanlines and of tiles, with one to three channels of every type, every compression and up to forty samples a pixel, a third damaged, read as for `tinyexr_parts` | 90,000 deep files (264 megabytes) read identically; the 1,006 readings that depended on memory outside the file or stopped tinyexr refused by the port |
| `tinyexr_zfp_write` | Made-up images for ZFP compression, with settings at a fixed rate, a precision or a tolerance, among them zero, negative, infinite and not-a-number values, and now and then missing or malformed | 160,000 images: 114,800 saved as identical files (595 megabytes) and 45,200 refused with the same message |
| `tinyexr_zfp_read` | The files tinyexr saved of those, a third damaged and some with their ZFP settings changed | 160,000 files (505 megabytes), of which 107,000 had a header tinyexr takes: 78,600 images read and 28,300 refused, each with the same code and message |

The C++ was built by gcc and clang, and run under gcc's address sanitizer for half of
`tinyexr_read`; the larger runs were repeated with Rust's overflow checks enabled. In addition,
files written by the port were read by ffmpeg, whose `.exr` reader is its own, to the same
floats, bit for bit, and files ffmpeg wrote read by the port to the same.

`cargo test` needs no C compiler: the tests check the port against files and hashes recorded
from tinyexr (`tests/data`), and read badly damaged files to check that each is read or refused
without a panic.

## Licence

As tinyexr is: BSD 3-Clause ([LICENSE](LICENSE), which keeps tinyexr's copyright notice and
that of the OpenEXR code it contains).
