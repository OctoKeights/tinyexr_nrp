# tinyexr_nrp

**OpenEXR files read and written in pure, safe Rust, bit-identical to tinyexr.**

**Overview** · [Guide](docs/guide.md) · [Verification](docs/verification.md) · [Performance](docs/performance.md)

---

tinyexr_nrp is a native Rust port of Syoyo Fujita's
[tinyexr](https://github.com/syoyo/tinyexr) v3.4.0. It writes the `.exr` files tinyexr
writes, byte for byte, reads the values tinyexr reads, and refuses the files tinyexr refuses,
with the same code and message. It also reads and writes DWAA and DWAB, which tinyexr does
not support, as OpenEXR does.

## Highlights

- **Every compression.** None, RLE, ZIPS, ZIP, PIZ, PXR24, B44 and B44A as tinyexr has them;
  DWAA and DWAB as OpenEXR 3.5 has them
  ([openexr_nrp](https://github.com/OctoKeights/openexr_nrp)); and the ZFP compression
  tinyexr adds with `TINYEXR_USE_ZFP` ([zfp_nrp](https://github.com/OctoKeights/zfp_nrp)).

- **Every layout.** Scanlines and tiles with mip and rip levels, files of several parts, and
  deep images (read only); halves, floats and 32-bit integers.

- **Damaged files refused safely.** Where tinyexr would read past the end of a damaged file
  or stop, tinyexr_nrp refuses it.

- **Its own dependencies only:** [miniz_nrp](https://github.com/OctoKeights/miniz_nrp) for
  ZIP, openexr_nrp and zfp_nrp. No C compiler, no build script.

- **No `unsafe` code.** Enforced by `#![forbid(unsafe_code)]`.

## Installation

```toml
[dependencies]
tinyexr_nrp = "0.2"
```

## Quick start

```rust
use tinyexr_nrp::{load_rgba, save_floats};

// An image of 8 by 4 pixels of RGBA floats.
let (width, height) = (8, 4);
let rgba: Vec<f32> = (0..width * height * 4).map(|i| i as f32 / 16.0).collect();

// Saved as floats (true would store halves), as tinyexr's SaveEXRToMemory writes it.
let file = save_floats(&rgba, width, height, 4, false).unwrap();
assert_eq!(&file[..4], &[0x76, 0x2f, 0x31, 0x01]);

// Read back as RGBA floats.
let image = load_rgba(&file).unwrap();
assert_eq!((image.width, image.height), (width, height));
assert_eq!(image.data, rgba);
```

`parse_version`, `parse_header` and `load_image` give the header and then every channel as
it is stored, or every tile of every level; `save_image` writes an image the same way. The
[Guide](docs/guide.md) covers these, files of several parts, deep images and the
compressions' settings.

## Scope

`tinyexr.h` as built without threads and with miniz for its ZIP compression: its readers and
writers of one part and of several, of scanlines and of tiles; its reader of deep images;
and every compression it has, ZFP included. Files are read from and written to memory.

## Licence

BSD 3-Clause, as tinyexr is; see [`LICENSE`](LICENSE), which keeps tinyexr's copyright
notices and those of the OpenEXR code it contains.

Bug reports and contributions are welcome, above all any file that tinyexr_nrp reads or
writes differently from tinyexr.
