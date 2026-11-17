# Performance

[Overview](../README.md) · [Guide](guide.md) · [Verification](verification.md) · **Performance**

---

`examples/bench.rs` saves and loads a synthetic 1024 by 768 image of half-float A, B, G and R
channels (smooth, with a little noise, like a photograph) with each compression, and
reports the best of five runs:

```sh
cargo run --release --example bench
cargo run --release --example bench piz 9   # one compression, nine runs
```

## Throughput

Measured on one core of an Intel Xeon at 2.1 GHz, built for generic x86-64, in megapixels
per second:

| Compression | Save | Load |
|---|---|---|
| None | 695 | 1,062 |
| RLE | 67.1 | 129 |
| ZIPS | 10.7 | 31.8 |
| ZIP | 8.6 | 28.2 |
| PIZ | 18.1 | 18.7 |
| PXR24 | 2.4 | 28.5 |
| B44 | 58.7 | 99.4 |
| DWAA | 15.4 | 27.6 |

## Compared with 0.1

Version 0.2 rewrote the crate's interface and the way channels are copied, and moved to the
0.2 releases of miniz_nrp and openexr_nrp. Measured alternately with 0.1 on the same
machine:

| Compression | Save | Load |
|---|---|---|
| None | 200% faster | 1,136% faster |
| RLE | 9% faster | 64% faster |
| ZIPS | 12% faster | 48% faster |
| ZIP | 9% faster | 30% faster |
| PIZ | 6% faster | 6% faster |
| PXR24 | 24% faster | 38% faster |
| B44 | 14% faster | 47% faster |
| DWAA | 19% faster | 46% faster |

Uncompressed files gain most: their values were stored one at a time, through a check of the
channel's type for each, and are now copied a line at a time. Every other compression gains
the same in the copying, and ZIP, ZIPS, PXR24 and DWAA from the faster compressors and
decompressors of miniz_nrp and openexr_nrp.
