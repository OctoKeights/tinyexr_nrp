//! The ZIP and RLE compression of OpenEXR: a port of `CompressZip`, `DecompressZip`,
//! `CompressRle` and `DecompressRle` of tinyexr v3.4.0 (`tinyexr.h`), built with miniz.
//!
//! Both reorder the bytes of a block first (every other byte to the front, the rest behind)
//! and store each byte as its difference from the one before, which is what makes smooth
//! pictures compress; a block that does not come out smaller is stored as it is.

use alloc::vec::Vec;
use miniz_nrp as miniz;

/// Splits `src` into even-indexed and odd-indexed halves in `tmp_buf` and applies OpenEXR's
/// `+128` delta predictor in SIMD-friendly chunks.
pub(crate) fn reorder_and_predict_into(src: &[u8], tmp_buf: &mut Vec<u8>) {
    let src_size = src.len();
    tmp_buf.clear();
    tmp_buf.resize(src_size, 0);
    if src_size == 0 {
        return;
    }

    // Reorder pixel bytes into even and odd halves (auto-vectorizable 16-pair blocks).
    {
        let half = src_size.div_ceil(2);
        let (evens, odds) = tmp_buf.split_at_mut(half);
        let (pairs, rem) = src.as_chunks::<2>();
        let (pair_blocks, pair_tail) = pairs.as_chunks::<16>();
        let (even_blocks, even_tail) = evens[..pairs.len()].as_chunks_mut::<16>();
        let (odd_blocks, odd_tail) = odds.as_chunks_mut::<16>();
        for ((eb, ob), pb) in even_blocks.iter_mut().zip(odd_blocks).zip(pair_blocks) {
            for i in 0..16 {
                eb[i] = pb[i][0];
                ob[i] = pb[i][1];
            }
        }
        for ((e, o), &[a, b]) in even_tail.iter_mut().zip(odd_tail).zip(pair_tail) {
            *e = a;
            *o = b;
        }
        if let &[last] = rem {
            evens[pairs.len()] = last;
        }
    }

    // Delta predictor (`d = cur - prev + 128`) in 32-byte register blocks with no loop-carried
    // dependency inside each block.
    if src_size > 1 {
        let mut prev = tmp_buf[0];
        let (blocks, tail) = tmp_buf[1..].as_chunks_mut::<32>();
        for block in blocks {
            let orig = *block;
            block[0] = orig[0].wrapping_sub(prev).wrapping_add(128);
            for i in 1..32 {
                block[i] = orig[i].wrapping_sub(orig[i - 1]).wrapping_add(128);
            }
            prev = orig[31];
        }
        for byte in tail {
            let orig = *byte;
            *byte = orig.wrapping_sub(prev).wrapping_add(128);
            prev = orig;
        }
    }
}

/// The reordering and the differences that both compressors start with.
#[cfg(test)]
fn reorder_and_predict(src: &[u8]) -> Vec<u8> {
    let mut tmp_buf = Vec::new();
    reorder_and_predict_into(src, &mut tmp_buf);
    tmp_buf
}

/// What both decompressors end with: the differences summed up, and the bytes put back in
/// their order, into `dst`.
pub(crate) fn unpredict_and_reorder(dst: &mut [u8], tmp_buf: &mut [u8]) {
    let size = tmp_buf.len();
    if size == 0 {
        return;
    }

    // Prefix-sum unpredictor.
    let mut acc = tmp_buf[0];
    for byte in &mut tmp_buf[1..] {
        acc = acc.wrapping_add(*byte).wrapping_sub(128);
        *byte = acc;
    }

    // Interleave even and odd halves back into `dst` (auto-vectorizable 16-pair blocks).
    let half = size.div_ceil(2);
    let (evens, odds) = tmp_buf.split_at(half);
    let (pairs, rem) = dst[..size].as_chunks_mut::<2>();
    let num_pairs = pairs.len();
    let (pair_blocks, pair_tail) = pairs.as_chunks_mut::<16>();
    let (even_blocks, even_tail) = evens[..num_pairs].as_chunks::<16>();
    let (odd_blocks, odd_tail) = odds.as_chunks::<16>();
    for (pb, (eb, ob)) in pair_blocks
        .iter_mut()
        .zip(even_blocks.iter().zip(odd_blocks))
    {
        for i in 0..16 {
            pb[i] = [eb[i], ob[i]];
        }
    }
    for (pair, (&e, &o)) in pair_tail.iter_mut().zip(even_tail.iter().zip(odd_tail)) {
        *pair = [e, o];
    }
    if let [last] = rem {
        *last = evens[num_pairs];
    }
}

/// `CompressZip` appending directly into `out_data`, reusing `tmp_buf` and `compressor` across
/// chunks.
pub(crate) fn compress_zip_into(
    src: &[u8],
    tmp_buf: &mut Vec<u8>,
    compressor: &mut miniz::Compressor,
    out_data: &mut Vec<u8>,
) {
    reorder_and_predict_into(src, tmp_buf);
    compressor.reset_with(&miniz::Options::default());
    compressor
        .compress(tmp_buf, miniz::Flush::Finish)
        .expect("freshly reset compressor");
    let compressed = compressor.output();
    // Use uncompressed data when compressed data is larger than uncompressed (Issue 40).
    if compressed.len() >= src.len() {
        out_data.extend_from_slice(src);
    } else {
        out_data.extend_from_slice(compressed);
    }
}

/// `DecompressZip` reusing `tmp_buf` across chunks.
pub(crate) fn decompress_zip_with_scratch(
    dst: &mut [u8],
    src: &[u8],
    tmp_buf: &mut Vec<u8>,
) -> Option<usize> {
    if dst.len() == src.len() {
        // Data is not compressed (Issue 40).
        dst.copy_from_slice(src);
        return Some(src.len());
    }
    miniz::decompress_into(src, miniz::Format::Zlib, dst.len(), tmp_buf).ok()?;
    unpredict_and_reorder(dst, tmp_buf);
    Some(tmp_buf.len())
}

/// `DecompressZip`: a block of as many bytes as `dst` holds into `dst`. The answer is how many
/// it held: a stream that holds fewer is taken, as the original takes it, and the rest of
/// `dst` is left as it was.
pub(crate) fn decompress_zip(dst: &mut [u8], src: &[u8]) -> Option<usize> {
    let mut tmp_buf = Vec::new();
    decompress_zip_with_scratch(dst, src, &mut tmp_buf)
}

const MIN_RUN_LENGTH: usize = 3;
const MAX_RUN_LENGTH: usize = 127;

/// `rleCompress`: runs of a byte as a count and the byte, and what lies between as a
/// negative count and the bytes. The answer is the size of what was written to `out`, which
/// has room for half as much again as `input`.
fn rle_compress(input: &[u8], out: &mut [u8]) -> usize {
    let in_end = input.len();
    let mut run_start = 0usize;
    let mut run_end = 1usize;
    let mut out_write = 0usize;

    while run_start < in_end {
        while run_end < in_end
            && input[run_start] == input[run_end]
            && (run_end - run_start) < MAX_RUN_LENGTH + 1
        {
            run_end += 1;
        }

        if run_end - run_start >= MIN_RUN_LENGTH {
            // Compressible run.
            out[out_write] = ((run_end - run_start) - 1) as u8;
            out_write += 1;
            out[out_write] = input[run_start];
            out_write += 1;
            run_start = run_end;
        } else {
            // Uncompressible run.
            while run_end < in_end
                && ((run_end + 1 >= in_end || input[run_end] != input[run_end + 1])
                    || (run_end + 2 >= in_end || input[run_end + 1] != input[run_end + 2]))
                && run_end - run_start < MAX_RUN_LENGTH
            {
                run_end += 1;
            }

            out[out_write] = ((run_start as isize) - (run_end as isize)) as u8;
            out_write += 1;

            let literal_len = run_end - run_start;
            out[out_write..out_write + literal_len].copy_from_slice(&input[run_start..run_end]);
            out_write += literal_len;
            run_start = run_end;
        }

        run_end += 1;
    }

    out_write
}

/// `rleUncompress`: returns the size of what was written to `out`, or `None` if the data
/// does not fit `max_length` bytes or runs past its end. Uses `usize` arithmetic so blocks
/// larger than `i32::MAX` never wrap negative.
fn rle_uncompress(input: &[u8], mut max_length: usize, out: &mut [u8]) -> Option<usize> {
    let mut i = 0usize;
    let mut o = 0usize;

    while i < input.len() {
        let cmd = input[i] as i8;
        i += 1;
        if cmd < 0 {
            let count = (-i32::from(cmd)) as usize;
            if input.len() - i < count || max_length < count {
                return None;
            }
            max_length -= count;
            out[o..o + count].copy_from_slice(&input[i..i + count]);
            o += count;
            i += count;
        } else {
            let run = (cmd as usize) + 1;
            if i >= input.len() || max_length < run {
                return None;
            }
            max_length -= run;
            out[o..o + run].fill(input[i]);
            o += run;
            i += 1;
        }
    }

    Some(o)
}

/// `CompressRle` appending into `out_data`, reusing `tmp_buf` and `rle_buf`. Returns `false`
/// where the original fails (a block of no bytes).
pub(crate) fn compress_rle_into(
    src: &[u8],
    tmp_buf: &mut Vec<u8>,
    rle_buf: &mut Vec<u8>,
    out_data: &mut Vec<u8>,
) -> bool {
    reorder_and_predict_into(src, tmp_buf);
    let cap = src.len() + src.len() / 2 + 2;
    rle_buf.clear();
    rle_buf.resize(cap, 0);
    let out_size = rle_compress(tmp_buf, rle_buf);
    if out_size == 0 {
        return false;
    }
    if out_size >= src.len() {
        out_data.extend_from_slice(src);
    } else {
        out_data.extend_from_slice(&rle_buf[..out_size]);
    }
    true
}

/// `CompressRle`: a block run-length coded, or as it is if that is not smaller. `None` where
/// the original fails (a block of no bytes).
#[allow(dead_code)]
pub(crate) fn compress_rle(src: &[u8]) -> Option<Vec<u8>> {
    let mut tmp_buf = Vec::new();
    let mut dst = Vec::new();
    let mut out = Vec::new();
    compress_rle_into(src, &mut tmp_buf, &mut dst, &mut out).then_some(out)
}

/// `DecompressRle` reusing `tmp_buf` across chunks.
pub(crate) fn decompress_rle_with_scratch(
    dst: &mut [u8],
    uncompressed_size: usize,
    src: &[u8],
    tmp_buf: &mut Vec<u8>,
) -> bool {
    if uncompressed_size == src.len() {
        // Data is not compressed (Issue 40).
        dst[..src.len()].copy_from_slice(src);
        return true;
    }

    // Workaround for issue #112.
    if src.len() <= 2 {
        return false;
    }

    tmp_buf.clear();
    tmp_buf.resize(uncompressed_size, 0);
    let Some(written) = rle_uncompress(src, uncompressed_size, tmp_buf) else {
        return false;
    };
    if written != uncompressed_size {
        return false;
    }
    unpredict_and_reorder(dst, tmp_buf);
    true
}

/// `DecompressRle`: a block of `uncompressed_size` bytes into `dst`.
pub(crate) fn decompress_rle(dst: &mut [u8], uncompressed_size: usize, src: &[u8]) -> bool {
    let mut tmp_buf = Vec::new();
    decompress_rle_with_scratch(dst, uncompressed_size, src, &mut tmp_buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn vectorized_reorder_and_predict_round_trips_all_tail_lengths() {
        for len in [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 513] {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            let mut packed = reorder_and_predict(&data);
            let mut back = vec![0u8; len];
            unpredict_and_reorder(&mut back, &mut packed);
            assert_eq!(back, data, "len {len}");
        }
    }
}
