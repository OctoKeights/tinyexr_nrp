//! The ZIP and RLE compression of OpenEXR: a port of `CompressZip`, `DecompressZip`,
//! `CompressRle` and `DecompressRle` of tinyexr v3.4.0 (`tinyexr.h`), built with miniz.
//!
//! Both reorder the bytes of a block first (every other byte to the front, the rest behind)
//! and store each byte as its difference from the one before, which is what makes smooth
//! pictures compress; a block that does not come out smaller is stored as it is.

use miniz_nrp as miniz;

/// The reordering and the differences that both compressors start with.
fn reorder_and_predict(src: &[u8]) -> Vec<u8> {
    let src_size = src.len();
    let mut tmp_buf = vec![0u8; src_size];

    // Reorder the pixel data.
    {
        let mut t1 = 0usize;
        let mut t2 = (src_size + 1) / 2;
        let mut s = 0usize;
        loop {
            if s < src_size {
                tmp_buf[t1] = src[s];
                t1 += 1;
                s += 1;
            } else {
                break;
            }
            if s < src_size {
                tmp_buf[t2] = src[s];
                t2 += 1;
                s += 1;
            } else {
                break;
            }
        }
    }

    // Predictor.
    if src_size > 0 {
        let mut p = tmp_buf[0] as i32;
        for byte in &mut tmp_buf[1..] {
            let d = *byte as i32 - p + (128 + 256);
            p = *byte as i32;
            *byte = d as u8;
        }
    }
    tmp_buf
}

/// What both decompressors end with: the differences summed up, and the bytes put back in
/// their order, into `dst`.
pub(crate) fn unpredict_and_reorder(dst: &mut [u8], tmp_buf: &mut [u8]) {
    let size = tmp_buf.len();
    // Predictor.
    for t in 1..size {
        let d = tmp_buf[t - 1] as i32 + tmp_buf[t] as i32 - 128;
        tmp_buf[t] = d as u8;
    }

    // Reorder the pixel data.
    {
        let mut t1 = 0usize;
        let mut t2 = (size + 1) / 2;
        let mut s = 0usize;
        loop {
            if s < size {
                dst[s] = tmp_buf[t1];
                s += 1;
                t1 += 1;
            } else {
                break;
            }
            if s < size {
                dst[s] = tmp_buf[t2];
                s += 1;
                t2 += 1;
            } else {
                break;
            }
        }
    }
}

/// `CompressZip`: a block as a zlib stream at miniz's default level, or as it is if that is
/// not smaller.
pub(crate) fn compress_zip(src: &[u8]) -> Vec<u8> {
    let tmp_buf = reorder_and_predict(src);
    let compressed = miniz::compress(&tmp_buf, miniz::Level::DEFAULT);

    // Use uncompressed data when compressed data is larger than uncompressed.
    // (Issue 40)
    if compressed.len() >= src.len() {
        return src.to_vec();
    }
    compressed
}

/// `DecompressZip`: a block of `*uncompressed_size` bytes into `dst`. A stream that holds
/// fewer bytes than that is taken, as the original takes it: `*uncompressed_size` then says
/// how many, and the rest of `dst` is left as it was.
pub(crate) fn decompress_zip(dst: &mut [u8], uncompressed_size: &mut usize, src: &[u8]) -> bool {
    if *uncompressed_size == src.len() {
        // Data is not compressed (Issue 40).
        dst[..src.len()].copy_from_slice(src);
        return true;
    }
    let mut tmp_buf = match miniz::decompress(src, *uncompressed_size) {
        Ok(tmp_buf) => tmp_buf,
        Err(_) => return false,
    };
    *uncompressed_size = tmp_buf.len();
    unpredict_and_reorder(dst, &mut tmp_buf);
    true
}

const MIN_RUN_LENGTH: isize = 3;
const MAX_RUN_LENGTH: isize = 127;

/// `rleCompress`: runs of a byte as a count and the byte, and what lies between as a
/// negative count and the bytes. The answer is the size of what was written to `out`, which
/// has room for half as much again as `input`.
fn rle_compress(input: &[u8], out: &mut [u8]) -> usize {
    let in_end = input.len() as isize;
    let mut run_start = 0isize;
    let mut run_end = 1isize;
    let mut out_write = 0usize;
    let at = |i: isize| input[i as usize];

    while run_start < in_end {
        while run_end < in_end
            && at(run_start) == at(run_end)
            && run_end - run_start - 1 < MAX_RUN_LENGTH
        {
            run_end += 1;
        }

        if run_end - run_start >= MIN_RUN_LENGTH {
            // Compressible run.
            out[out_write] = ((run_end - run_start) as i8 as i32 - 1) as u8;
            out_write += 1;
            out[out_write] = at(run_start);
            out_write += 1;
            run_start = run_end;
        } else {
            // Uncompressable run.
            while run_end < in_end
                && ((run_end + 1 >= in_end || at(run_end) != at(run_end + 1))
                    || (run_end + 2 >= in_end || at(run_end + 1) != at(run_end + 2)))
                && run_end - run_start < MAX_RUN_LENGTH
            {
                run_end += 1;
            }

            out[out_write] = (run_start - run_end) as u8;
            out_write += 1;

            while run_start < run_end {
                out[out_write] = at(run_start);
                out_write += 1;
                run_start += 1;
            }
        }

        run_end += 1;
    }

    out_write
}

/// `rleUncompress`: the answer is the size of what was written to `out`, or 0 if the data
/// does not fit `max_length` bytes or runs past its end.
fn rle_uncompress(input: &[u8], mut max_length: i32, out: &mut [u8]) -> i32 {
    let mut in_length = input.len() as i32;
    let mut i = 0usize;
    let mut o = 0usize;

    while in_length > 0 {
        if (input[i] as i8) < 0 {
            let count = -(input[i] as i8 as i32);
            i += 1;
            in_length -= count + 1;

            // Fewer than count bytes left of the input, or of the output.
            max_length -= count;
            if 0 > max_length || in_length < 0 {
                return 0;
            }

            let count = count as usize;
            out[o..o + count].copy_from_slice(&input[i..i + count]);
            o += count;
            i += count;
        } else {
            let count = input[i] as i32;
            i += 1;
            in_length -= 2;

            max_length -= count + 1;
            if 0 > max_length || in_length < 0 {
                return 0;
            }

            let count = count as usize;
            out[o..o + count + 1].fill(input[i]);
            o += count + 1;
            i += 1;
        }
    }

    o as i32
}

/// `CompressRle`: a block run-length coded, or as it is if that is not smaller. `None` where
/// the original fails (a block of no bytes).
pub(crate) fn compress_rle(src: &[u8]) -> Option<Vec<u8>> {
    let tmp_buf = reorder_and_predict(src);
    // (The original writes into the caller's block, which has room for the worst case.)
    let mut dst = vec![0u8; src.len() + src.len() / 2 + 2];
    let out_size = rle_compress(&tmp_buf, &mut dst);
    if out_size == 0 {
        return None;
    }

    // Use uncompressed data when compressed data is larger than uncompressed.
    // (Issue 40)
    if out_size >= src.len() {
        return Some(src.to_vec());
    }
    dst.truncate(out_size);
    Some(dst)
}

/// `DecompressRle`: a block of `uncompressed_size` bytes into `dst`.
pub(crate) fn decompress_rle(dst: &mut [u8], uncompressed_size: usize, src: &[u8]) -> bool {
    if uncompressed_size == src.len() {
        // Data is not compressed (Issue 40).
        dst[..src.len()].copy_from_slice(src);
        return true;
    }

    // Workaround for issue #112.
    // TODO(syoyo): Add more robust out-of-bounds check in `rleUncompress`.
    if src.len() <= 2 {
        return false;
    }

    let mut tmp_buf = vec![0u8; uncompressed_size];
    let ret = rle_uncompress(src, uncompressed_size as i32, &mut tmp_buf);
    if ret != uncompressed_size as i32 {
        return false;
    }
    unpredict_and_reorder(dst, &mut tmp_buf);
    true
}
