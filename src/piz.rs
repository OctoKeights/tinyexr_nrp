//! The PIZ compression of OpenEXR: a port of `CompressPiz`, `DecompressPiz` and what is
//! under them in tinyexr v3.4.0 (`tinyexr.h`), which has them from OpenEXR's `ImfPizCompressor`,
//! `ImfWav` and `ImfHuf`.
//!
//! The 16-bit values of a block go through a table that closes the gaps between the values
//! that occur, then through a wavelet transform, and what comes out is Huffman coded.

// (The arithmetic is written as the original has it.)
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

/// `wenc14`: the wavelet's step for values under 2^14: an average and a difference.
#[inline]
fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let a_s = a as i16 as i32;
    let b_s = b as i16 as i32;

    let ms = ((a_s + b_s) >> 1) as i16;
    let ds = (a_s - b_s) as i16;

    (ms as u16, ds as u16)
}

/// `wdec14`.
#[inline]
fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let ls = l as i16 as i32;
    let hs = h as i16;

    let hi = hs as i32;
    let ai = ls + (hi & 1) + (hi >> 1);

    let a_s = ai as i16;
    let bs = (ai - hi) as i16;

    (a_s as u16, bs as u16)
}

const NBITS: i32 = 16;
const A_OFFSET: i32 = 1 << (NBITS - 1);
const M_OFFSET: i32 = 1 << (NBITS - 1);
const MOD_MASK: i32 = (1 << NBITS) - 1;

/// `wenc16`: the step for all 16 bits, with arithmetic modulo 2^16.
#[inline]
fn wenc16(a: u16, b: u16) -> (u16, u16) {
    let ao = (a as i32 + A_OFFSET) & MOD_MASK;
    let mut m = (ao + b as i32) >> 1;
    let mut d = ao - b as i32;

    if d < 0 {
        m = (m + M_OFFSET) & MOD_MASK;
    }

    d &= MOD_MASK;

    (m as u16, d as u16)
}

/// `wdec16`.
#[inline]
fn wdec16(l: u16, h: u16) -> (u16, u16) {
    let m = l as i32;
    let d = h as i32;
    let bb = (m - (d >> 1)) & MOD_MASK;
    let aa = (d + bb - A_OFFSET) & MOD_MASK;
    (aa as u16, bb as u16)
}

/// `wav2Encode`: the two-dimensional wavelet transform of `nx` by `ny` values in place,
/// `ox` and `oy` apart in `data`. `mx` is the largest value there is.
pub(crate) fn wav2_encode(data: &mut [u16], nx: i32, ox: i32, ny: i32, oy: i32, mx: u16) {
    let w14 = (mx as i32) < (1 << 14);
    let enc = |a: u16, b: u16| if w14 { wenc14(a, b) } else { wenc16(a, b) };

    let n = if nx > ny { ny } else { nx };
    let mut p = 1i32; // == 1 << level
    let mut p2 = 2i32; // == 1 << (level + 1)
    let (ox, oy) = (ox as isize, oy as isize);

    // Hierarchical loop on smaller dimension n.
    while p2 <= n {
        let mut py = 0isize;
        let ey = oy * (ny - p2) as isize;
        let oy1 = oy * p as isize;
        let oy2 = oy * p2 as isize;
        let ox1 = ox * p as isize;
        let ox2 = ox * p2 as isize;

        // Y loop.
        while py <= ey {
            let mut px = py;
            let ex = py + ox * (nx - p2) as isize;

            // X loop.
            while px <= ex {
                let p01 = px + ox1;
                let p10 = px + oy1;
                let p11 = p10 + ox1;

                // 2D wavelet encoding.
                let (i00, i01) = enc(data[px as usize], data[p01 as usize]);
                let (i10, i11) = enc(data[p10 as usize], data[p11 as usize]);
                (data[px as usize], data[p10 as usize]) = enc(i00, i10);
                (data[p01 as usize], data[p11 as usize]) = enc(i01, i11);
                px += ox2;
            }

            // Encode (1D) odd column (still in Y loop).
            if nx & p != 0 {
                let p10 = px + oy1;
                let (i00, h) = enc(data[px as usize], data[p10 as usize]);
                data[p10 as usize] = h;
                data[px as usize] = i00;
            }
            py += oy2;
        }

        // Encode (1D) odd line (must loop in X).
        if ny & p != 0 {
            let mut px = py;
            let ex = py + ox * (nx - p2) as isize;

            while px <= ex {
                let p01 = px + ox1;
                let (i00, h) = enc(data[px as usize], data[p01 as usize]);
                data[p01 as usize] = h;
                data[px as usize] = i00;
                px += ox2;
            }
        }

        // Next level.
        p = p2;
        p2 <<= 1;
    }
}

/// `wav2Decode`: the transform undone.
pub(crate) fn wav2_decode(data: &mut [u16], nx: i32, ox: i32, ny: i32, oy: i32, mx: u16) {
    let w14 = (mx as i32) < (1 << 14);
    let dec = |l: u16, h: u16| if w14 { wdec14(l, h) } else { wdec16(l, h) };

    let n = if nx > ny { ny } else { nx };
    let mut p = 1i32;
    let (ox, oy) = (ox as isize, oy as isize);

    // Search max level.
    while p <= n {
        p <<= 1;
    }

    p >>= 1;
    let mut p2 = p;
    p >>= 1;

    // Hierarchical loop on smaller dimension n.
    while p >= 1 {
        let mut py = 0isize;
        let ey = oy * (ny - p2) as isize;
        let oy1 = oy * p as isize;
        let oy2 = oy * p2 as isize;
        let ox1 = ox * p as isize;
        let ox2 = ox * p2 as isize;

        // Y loop.
        while py <= ey {
            let mut px = py;
            let ex = py + ox * (nx - p2) as isize;

            // X loop.
            while px <= ex {
                let p01 = px + ox1;
                let p10 = px + oy1;
                let p11 = p10 + ox1;

                // 2D wavelet decoding.
                let (i00, i10) = dec(data[px as usize], data[p10 as usize]);
                let (i01, i11) = dec(data[p01 as usize], data[p11 as usize]);
                (data[px as usize], data[p01 as usize]) = dec(i00, i01);
                (data[p10 as usize], data[p11 as usize]) = dec(i10, i11);
                px += ox2;
            }

            // Decode (1D) odd column (still in Y loop).
            if nx & p != 0 {
                let p10 = px + oy1;
                let (i00, b) = dec(data[px as usize], data[p10 as usize]);
                data[p10 as usize] = b;
                data[px as usize] = i00;
            }
            py += oy2;
        }

        // Decode (1D) odd line (must loop in X).
        if ny & p != 0 {
            let mut px = py;
            let ex = py + ox * (nx - p2) as isize;

            while px <= ex {
                let p01 = px + ox1;
                let (i00, b) = dec(data[px as usize], data[p01 as usize]);
                data[p01 as usize] = b;
                data[px as usize] = i00;
                px += ox2;
            }
        }

        // Next level.
        p2 = p;
        p >>= 1;
    }
}

// The Huffman coder (OpenEXR's `ImfHuf`).

const HUF_ENCBITS: i32 = 16; // literal (value) bit length
const HUF_DECBITS: i32 = 14; // decoding bit size (>= 8)

const HUF_ENCSIZE: usize = (1 << HUF_ENCBITS) + 1; // encoding table size
const HUF_DECSIZE: usize = 1 << HUF_DECBITS; // decoding table size
const HUF_DECMASK: u64 = HUF_DECSIZE as u64 - 1;

/// `HufDec`: an entry of the decoding table: a short code's length and value, or the list of
/// the long codes that start with these bits.
#[derive(Clone, Default)]
struct HufDec {
    len: u32,
    lit: u32,
    p: Option<Vec<u32>>,
}

#[inline]
fn huf_length(code: i64) -> i64 {
    code & 63
}

#[inline]
fn huf_code(code: i64) -> i64 {
    code >> 6
}

/// Where the bits of a stream being written are kept (the original's `c`, `lc` and `out`).
struct BitWriter<'a> {
    c: u64,
    lc: i32,
    out: &'a mut [u8],
    pos: usize,
}

impl BitWriter<'_> {
    /// `outputBits`.
    #[inline]
    fn output_bits(&mut self, n_bits: i32, bits: i64) {
        self.c = self.c.wrapping_shl(n_bits as u32);
        self.lc += n_bits;

        self.c |= bits as u64;

        while self.lc >= 8 {
            self.lc -= 8;
            self.out[self.pos] = (self.c >> self.lc) as u8;
            self.pos += 1;
        }
    }

    /// `outputCode`.
    #[inline]
    fn output_code(&mut self, code: i64) {
        self.output_bits(huf_length(code) as i32, huf_code(code));
    }

    /// `sendCode`: a symbol and how many more times it comes, as a run where that is shorter.
    #[inline]
    fn send_code(&mut self, s_code: i64, mut run_count: i32, run_code: i64) {
        // Output a run of runCount instances of the symbol sCount. Output the symbols
        // explicitly, or if that is shorter, output the sCode symbol once followed by a
        // runCode symbol and runCount expressed as an 8-bit number.
        if huf_length(s_code) + huf_length(run_code) + 8 < huf_length(s_code) * run_count as i64 {
            self.output_code(s_code);
            self.output_code(run_code);
            self.output_bits(8, run_count as i64);
        } else {
            while run_count >= 0 {
                self.output_code(s_code);
                run_count -= 1;
            }
        }
    }
}

/// Where the bits of a stream being read are kept (the original's `c`, `lc` and `in`).
struct BitReader<'a> {
    c: u64,
    lc: i32,
    input: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    /// `getChar`. (A byte past the end of the data reads as 0: the original reads whatever
    /// is there, in the one place where it can ask for one.)
    #[inline]
    fn get_char(&mut self) {
        self.c = (self.c << 8) | self.input.get(self.pos).copied().unwrap_or(0) as u64;
        self.pos += 1;
        self.lc += 8;
    }

    /// `getBits`.
    #[inline]
    fn get_bits(&mut self, n_bits: i32) -> i64 {
        while self.lc < n_bits {
            self.get_char();
        }

        self.lc -= n_bits;
        ((self.c >> self.lc) & ((1u64 << n_bits) - 1)) as i64
    }
}

/// `hufCanonicalCodeTable`: the lengths in `hcode` become codes with their lengths in the
/// low six bits. Codes of one length are in the order of their symbols, and shorter codes
/// have higher values.
fn huf_canonical_code_table(hcode: &mut [i64]) {
    let mut n = [0i64; 59];

    // For each i from 0 through 58, count the number of different codes of length i, and
    // store the count in n[i].
    for i in 0..HUF_ENCSIZE {
        n[hcode[i] as usize] += 1;
    }

    // For each i from 58 through 1, compute the numerically lowest code with length i, and
    // store that code in n[i].
    let mut c = 0i64;

    for i in (1..=58).rev() {
        let nc = (c + n[i]) >> 1;
        n[i] = c;
        c = nc;
    }

    // hcode[i] contains the length, l, of the code for symbol i. Assign the next available
    // code of length l to the symbol and store both l and the code in hcode[i].
    for i in 0..HUF_ENCSIZE {
        let l = hcode[i] as i32;

        if l > 0 {
            hcode[i] = l as i64 | (n[l as usize] << 6);
            n[l as usize] += 1;
        }
    }
}

// The heap of the symbols' frequencies, least first, is `std::make_heap`, `std::pop_heap`
// and `std::push_heap` in the original, and which of two equal frequencies comes first
// decides the codes: the three are written here as GNU libstdc++ has them
// (`bits/stl_heap.h`), which is what the port is compared with. `greater` is the
// original's `FHeapCompare`.

/// `std::__push_heap`.
fn heap_push_at(
    heap: &mut [usize],
    mut hole_index: usize,
    top_index: usize,
    value: usize,
    greater: &dyn Fn(usize, usize) -> bool,
) {
    let mut parent = hole_index.wrapping_sub(1) / 2;
    while hole_index > top_index && greater(heap[parent], value) {
        heap[hole_index] = heap[parent];
        hole_index = parent;
        parent = hole_index.wrapping_sub(1) / 2;
    }
    heap[hole_index] = value;
}

/// `std::__adjust_heap`.
fn heap_adjust(
    heap: &mut [usize],
    mut hole_index: usize,
    len: usize,
    value: usize,
    greater: &dyn Fn(usize, usize) -> bool,
) {
    let top_index = hole_index;
    let mut second_child = hole_index;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if greater(heap[second_child], heap[second_child - 1]) {
            second_child -= 1;
        }
        heap[hole_index] = heap[second_child];
        hole_index = second_child;
    }
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        heap[hole_index] = heap[second_child - 1];
        hole_index = second_child - 1;
    }
    heap_push_at(heap, hole_index, top_index, value, greater);
}

/// `std::make_heap` of `heap[..len]`.
fn make_heap(heap: &mut [usize], len: usize, greater: &dyn Fn(usize, usize) -> bool) {
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = heap[parent];
        heap_adjust(heap, parent, len, value, greater);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::pop_heap` of `heap[..len]`: the first goes to the end.
fn pop_heap(heap: &mut [usize], len: usize, greater: &dyn Fn(usize, usize) -> bool) {
    if len > 1 {
        let last = len - 1;
        let value = heap[last];
        heap[last] = heap[0];
        heap_adjust(heap, 0, last, value, greater);
    }
}

/// `std::push_heap` of `heap[..len]`: the last finds its place.
fn push_heap(heap: &mut [usize], len: usize, greater: &dyn Fn(usize, usize) -> bool) {
    let value = heap[len - 1];
    heap_push_at(heap, len - 1, 0, value, greater);
}

/// `hufBuildEncTable`: the frequencies in `frq` become the symbols' codes and lengths.
/// `im` and `i_m` come back as the lowest and highest symbols with a code; the highest is
/// one past the last value that occurs, and stands for a run.
fn huf_build_enc_table(frq: &mut [i64], im: &mut i32, i_m: &mut i32) -> bool {
    // This function assumes that when it is called, array frq indicates the frequency of
    // all possible symbols in the data that are to be Huffman-encoded. (frq[i] contains the
    // number of occurrences of symbol i in the data.)
    //
    // Find the lowest and highest indices where frq is nonzero, and add a pseudo-symbol,
    // with a frequency count of 1, to frq; it is used for run-length encoding.
    let mut hlink = vec![0i32; HUF_ENCSIZE];
    // (Places in `frq`, where the original has pointers into it.)
    let mut f_heap = vec![0usize; HUF_ENCSIZE];

    *im = 0;

    while frq[*im as usize] == 0 {
        *im += 1;
    }

    let mut nf = 0usize;

    for i in *im as usize..HUF_ENCSIZE {
        hlink[i] = i as i32;

        if frq[i] != 0 {
            f_heap[nf] = i;
            nf += 1;
            *i_m = i as i32;
        }
    }

    // Add a pseudo-symbol, with a frequency count of 1, to frq; adjust the fHeap and hlink
    // array accordingly.
    *i_m += 1;
    frq[*i_m as usize] = 1;
    f_heap[nf] = *i_m as usize;
    nf += 1;

    // Build an array, scode, such that scode[i] contains the number of bits assigned to
    // symbol i. Conceptually this is done by constructing a tree whose leaves are the
    // symbols with non-zero frequency: the two least frequent are merged over and over.
    {
        let frq_now = &*frq;
        make_heap(&mut f_heap, nf, &|a, b| frq_now[a] > frq_now[b]);
    }

    let mut scode = vec![0i64; HUF_ENCSIZE];

    while nf > 1 {
        // Find the indices, mm and m, of the two smallest non-zero frq values in fHeap,
        // add the smallest frq to the second-smallest frq, and remove the smallest frq
        // value from fHeap.
        let mm = f_heap[0];
        {
            let frq_now = &*frq;
            pop_heap(&mut f_heap, nf, &|a, b| frq_now[a] > frq_now[b]);
        }
        nf -= 1;

        let m = f_heap[0];
        {
            let frq_now = &*frq;
            pop_heap(&mut f_heap, nf, &|a, b| frq_now[a] > frq_now[b]);
        }

        frq[m] += frq[mm];
        {
            let frq_now = &*frq;
            push_heap(&mut f_heap, nf, &|a, b| frq_now[a] > frq_now[b]);
        }

        // The entries in scode are linked into lists with the entries in hlink serving as
        // "next" pointers and with the end of a list marked by hlink[j] == j.
        //
        // Traverse the lists that start at scode[m] and scode[mm]. For each element
        // visited, increment the length of the corresponding code by one bit. (If we
        // visit scode[j] during the traversal, then the code for symbol j becomes one bit
        // longer.)
        //
        // Merge the lists that start at scode[m] and scode[mm] into a single list that
        // starts at scode[m].

        // Add a bit to all codes in the first list.
        let mut j = m;
        loop {
            scode[j] += 1;

            if scode[j] > 58 {
                return false;
            }

            if hlink[j] as usize == j {
                // Merge the two lists.
                hlink[j] = mm as i32;
                break;
            }
            j = hlink[j] as usize;
        }

        // Add a bit to all codes in the second list.
        let mut j = mm;
        loop {
            scode[j] += 1;

            if scode[j] > 58 {
                return false;
            }

            if hlink[j] as usize == j {
                break;
            }
            j = hlink[j] as usize;
        }
    }

    // Build a canonical Huffman code table, replacing the code lengths in scode with
    // (code, code length) pairs. Copy the code table from scode into frq.
    huf_canonical_code_table(&mut scode);
    frq[..HUF_ENCSIZE].copy_from_slice(&scode);

    true
}

// Pack an encoding table: only code lengths, not actual codes, are stored; runs of zeroes
// are compressed as follows:
//
//    unpacked            packed
//    --------------------------------
//    1 zero              0       (6 bits)
//    2 zeroes            59
//    3 zeroes            60
//    4 zeroes            61
//    5 zeroes            62
//    n zeroes (6 or more)    63 n-6      (6 + 8 bits)
const SHORT_ZEROCODE_RUN: i32 = 59;
const LONG_ZEROCODE_RUN: i32 = 63;
const SHORTEST_LONG_RUN: i32 = 2 + LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN;
const LONGEST_LONG_RUN: i32 = 255 + SHORTEST_LONG_RUN;

/// `hufPackEncTable`: the lengths of the codes from `im` to `i_m`, written to `out`. The
/// answer is how many bytes that took.
fn huf_pack_enc_table(hcode: &[i64], mut im: i32, i_m: i32, out: &mut [u8]) -> usize {
    let mut w = BitWriter {
        c: 0,
        lc: 0,
        out,
        pos: 0,
    };

    while im <= i_m {
        let l = huf_length(hcode[im as usize]) as i32;

        if l == 0 {
            let mut zerun = 1;

            while im < i_m && zerun < LONGEST_LONG_RUN {
                if huf_length(hcode[im as usize + 1]) > 0 {
                    break;
                }
                im += 1;
                zerun += 1;
            }

            if zerun >= 2 {
                if zerun >= SHORTEST_LONG_RUN {
                    w.output_bits(6, LONG_ZEROCODE_RUN as i64);
                    w.output_bits(8, (zerun - SHORTEST_LONG_RUN) as i64);
                } else {
                    w.output_bits(6, (SHORT_ZEROCODE_RUN + zerun - 2) as i64);
                }
                im += 1;
                continue;
            }
        }

        w.output_bits(6, l as i64);
        im += 1;
    }

    if w.lc > 0 {
        w.out[w.pos] = (w.c << (8 - w.lc)) as u8;
        w.pos += 1;
    }

    w.pos
}

/// `hufUnpackEncTable`: the lengths of the codes from `im` to `i_m` out of the `ni` bytes
/// at `*pcode` in `data`, and from them the codes. `false`, with `*pcode` where it was and
/// `hcode` as far as it got, when the table does not fit.
fn huf_unpack_enc_table(
    data: &[u8],
    pcode: &mut usize,
    ni: i64,
    mut im: i32,
    i_m: i32,
    hcode: &mut [i64],
) -> bool {
    hcode[..HUF_ENCSIZE].fill(0);

    let mut r = BitReader {
        c: 0,
        lc: 0,
        input: data,
        pos: *pcode,
    };

    while im <= i_m {
        if (r.pos - *pcode) as i64 >= ni {
            return false;
        }

        let l = r.get_bits(6); // code length
        hcode[im as usize] = l;

        if l == LONG_ZEROCODE_RUN as i64 {
            if (r.pos - *pcode) as i64 > ni {
                return false;
            }

            let mut zerun = r.get_bits(8) as i32 + SHORTEST_LONG_RUN;

            if im + zerun > i_m + 1 {
                return false;
            }

            while zerun > 0 {
                hcode[im as usize] = 0;
                im += 1;
                zerun -= 1;
            }

            im -= 1;
        } else if l >= SHORT_ZEROCODE_RUN as i64 {
            let mut zerun = l as i32 - SHORT_ZEROCODE_RUN + 2;

            if im + zerun > i_m + 1 {
                return false;
            }

            while zerun > 0 {
                hcode[im as usize] = 0;
                im += 1;
                zerun -= 1;
            }

            im -= 1;
        }
        im += 1;
    }

    *pcode = r.pos;

    huf_canonical_code_table(hcode);

    true
}

/// `hufBuildDecTable`: the table that decodes the codes from `im` to `i_m`. `false`, with
/// the table as far as it got, when the codes are not a code.
fn huf_build_dec_table(hcode: &[i64], mut im: i32, i_m: i32, hdecod: &mut [HufDec]) -> bool {
    // Init hashtable & loop on all codes. Assumes that hufClearDecTable(hdecod) has already
    // been called.
    while im <= i_m {
        let c = huf_code(hcode[im as usize]);
        let l = huf_length(hcode[im as usize]) as i32;

        if c >> l != 0 {
            // Error: c is supposed to be an l-bit code, but c contains a value that is
            // greater than the largest l-bit number.
            return false;
        }

        if l > HUF_DECBITS {
            // Long code: add a secondary entry.
            let pl = &mut hdecod[(c >> (l - HUF_DECBITS)) as usize];

            if pl.len != 0 {
                // Error: a short code has already been stored in table entry *pl.
                return false;
            }

            pl.lit += 1;

            match &mut pl.p {
                Some(p) => p.push(im as u32),
                None => pl.p = Some(vec![im as u32]),
            }
        } else if l != 0 {
            // Short code: init all primary entries.
            let start = (c << (HUF_DECBITS - l)) as usize;

            for pl in &mut hdecod[start..start + (1usize << (HUF_DECBITS - l))] {
                if pl.len != 0 || pl.p.is_some() {
                    // Error: a short code or a long code has already been stored in table
                    // entry *pl.
                    return false;
                }

                pl.len = l as u32;
                pl.lit = im as u32;
            }
        }
        im += 1;
    }

    true
}

/// `hufEncode`: the values of `input`, coded, into `out`. The answer is the number of bits.
fn huf_encode(hcode: &[i64], input: &[u16], rlc: i32, out: &mut [u8]) -> i32 {
    let ni = input.len();
    let mut w = BitWriter {
        c: 0,
        lc: 0,
        out,
        pos: 0,
    };
    let mut s = input[0] as usize;
    let mut cs = 0;

    // Loop on input values.
    for i in 1..ni {
        // Count same values or send code.
        if s == input[i] as usize && cs < 255 {
            cs += 1;
        } else {
            w.send_code(hcode[s], cs, hcode[rlc as usize]);
            cs = 0;
        }

        s = input[i] as usize;
    }

    // Send remaining code.
    w.send_code(hcode[s], cs, hcode[rlc as usize]);

    if w.lc != 0 {
        w.out[w.pos] = (w.c << (8 - w.lc)) as u8;
    }

    (w.pos * 8) as i32 + w.lc
}

/// What `hufDecode` writes to: the values so far, and how many there may be.
struct DecodeOut<'a> {
    out: &'a mut [u16],
    pos: usize,
}

/// `getCode`: a decoded symbol to the output, or, for the run symbol, the last value again
/// as many times as the next eight bits say.
#[inline]
fn get_code(po: u32, rlc: i32, r: &mut BitReader, in_end: usize, o: &mut DecodeOut) -> bool {
    if po as i32 == rlc {
        if r.lc < 8 {
            // TinyEXR issue 78.
            if r.pos >= in_end {
                return false;
            }
            r.get_char();
        }

        r.lc -= 8;

        let cs = ((r.c >> r.lc) & 0xff) as usize;

        if o.pos + cs > o.out.len() {
            return false;
        }

        // Bounds check for safety. Issue 100.
        if o.pos < 1 {
            return false;
        }
        let s = o.out[o.pos - 1];

        for _ in 0..cs {
            o.out[o.pos] = s;
            o.pos += 1;
        }
    } else if o.pos < o.out.len() {
        o.out[o.pos] = po as u16;
        o.pos += 1;
    } else {
        return false;
    }
    true
}

/// `hufDecode`: `ni` bits at `start` of `data` into the values of `out`. `false`, with
/// `out` as far as it got, when the bits are not what the table codes.
fn huf_decode(
    hcode: &[i64],
    hdecod: &[HufDec],
    data: &[u8],
    start: usize,
    ni: i32,
    rlc: i32,
    out: &mut [u16],
) -> bool {
    let no = out.len();
    let mut o = DecodeOut { out, pos: 0 };
    let mut r = BitReader {
        c: 0,
        lc: 0,
        input: data,
        pos: start,
    };
    // `ie` is `in + (ni + 7) / 8`, which is before `in` for a negative number of bits.
    let ie = start as i64 + ((ni + 7) / 8) as i64;

    // Loop on input bytes.
    while (r.pos as i64) < ie {
        r.get_char();

        // Access decoding table.
        while r.lc >= HUF_DECBITS {
            let pl = &hdecod[((r.c >> (r.lc - HUF_DECBITS)) & HUF_DECMASK) as usize];

            if pl.len != 0 {
                // Get short code.
                r.lc -= pl.len as i32;
                if !get_code(pl.lit, rlc, &mut r, ie as usize, &mut o) {
                    return false;
                }
            } else {
                let Some(p) = &pl.p else {
                    return false; // wrong code
                };

                // Search long code.
                let mut j = 0;
                while j < pl.lit as usize {
                    let l = huf_length(hcode[p[j] as usize]) as i32;

                    while r.lc < l && (r.pos as i64) < ie {
                        // Get more bits.
                        r.get_char();
                    }

                    if r.lc >= l
                        && huf_code(hcode[p[j] as usize]) as u64
                            == ((r.c >> (r.lc - l)) & ((1u64 << l) - 1))
                    {
                        // Found: get long code.
                        r.lc -= l;
                        if !get_code(p[j], rlc, &mut r, ie as usize, &mut o) {
                            return false;
                        }
                        break;
                    }
                    j += 1;
                }

                if j == pl.lit as usize {
                    return false; // Not found.
                }
            }
        }
    }

    // Get remaining (short) codes.
    let i = (8 - ni) & 7;
    r.c = r.c.wrapping_shr(i as u32);
    r.lc -= i;

    while r.lc > 0 {
        let pl = &hdecod[(r.c.wrapping_shl((HUF_DECBITS - r.lc) as u32) & HUF_DECMASK) as usize];

        if pl.len != 0 {
            r.lc -= pl.len as i32;
            if !get_code(pl.lit, rlc, &mut r, ie.max(0) as usize, &mut o) {
                return false;
            }
        } else {
            return false; // wrong (long) code
        }
    }

    o.pos == no
}

/// `hufCompress`: the 16-bit values of `raw` into `compressed`: a header of twenty bytes,
/// the table of code lengths, then the codes. The answer is how many bytes that took.
pub(crate) fn huf_compress(raw: &[u16], compressed: &mut [u8]) -> usize {
    if raw.is_empty() {
        return 0;
    }

    let mut freq = vec![0i64; HUF_ENCSIZE];

    // `countFrequencies`.
    for &value in raw {
        freq[value as usize] += 1;
    }

    let mut im = 0;
    let mut i_m = 0;
    huf_build_enc_table(&mut freq, &mut im, &mut i_m);

    let table_start = 20usize;
    let table_length = huf_pack_enc_table(&freq, im, i_m, &mut compressed[table_start..]);

    let data_start = table_start + table_length;
    let n_bits = huf_encode(&freq, raw, i_m, &mut compressed[data_start..]);
    let data_length = ((n_bits + 7) / 8) as usize;

    compressed[0..4].copy_from_slice(&(im as u32).to_le_bytes());
    compressed[4..8].copy_from_slice(&(i_m as u32).to_le_bytes());
    compressed[8..12].copy_from_slice(&(table_length as u32).to_le_bytes());
    compressed[12..16].copy_from_slice(&(n_bits as u32).to_le_bytes());
    compressed[16..20].copy_from_slice(&0u32.to_le_bytes()); // room for future extensions

    data_start + data_length
}

/// `hufUncompress`: the values of `raw` out of the `n_compressed` bytes at the start of
/// `compressed`, which goes on to the end of the file (a damaged stream can have the original
/// read a little past its own end, into what follows it there). As the original, it only
/// says `false` for data that cannot be a stream at all; where the table or the codes turn
/// out damaged further in, `raw` holds what was decoded up to there and the answer is
/// `true`.
pub(crate) fn huf_uncompress(compressed: &[u8], n_compressed: i64, raw: &mut [u16]) -> bool {
    if n_compressed == 0 {
        return false;
    }

    // (Past the end of the file the original reads what is not its own; here that is 0.)
    let read_uint = |at: usize| {
        let byte = |i: usize| compressed.get(at + i).copied().unwrap_or(0) as u32;
        (byte(0) | (byte(1) << 8) | (byte(2) << 16) | (byte(3) << 24)) as i32
    };
    let im = read_uint(0);
    let i_m = read_uint(4);
    // let tableLength = readUInt(compressed + 8);
    let n_bits = read_uint(12);

    if im < 0 || im >= HUF_ENCSIZE as i32 || i_m < 0 || i_m >= HUF_ENCSIZE as i32 {
        return false;
    }

    let mut ptr = 20usize;

    {
        let mut freq = vec![0i64; HUF_ENCSIZE];
        let mut hdec = vec![HufDec::default(); HUF_DECSIZE];

        // (None of the three answers is looked at by the original.)
        let table_room = n_compressed - ptr as i64;
        huf_unpack_enc_table(compressed, &mut ptr, table_room, im, i_m, &mut freq);

        {
            if n_bits as i64 > 8 * (n_compressed - ptr as i64) {
                return false;
            }

            huf_build_dec_table(&freq, im, i_m, &mut hdec);
            huf_decode(&freq, &hdec, compressed, ptr, n_bits, i_m, raw);
        }
    }

    true
}

// The PIZ compressor itself (OpenEXR's `ImfPizCompressor`).

const USHORT_RANGE: usize = 1 << 16;
const BITMAP_SIZE: usize = USHORT_RANGE >> 3;

/// `bitmapFromData`: a bit for every value that occurs, and the first and last bytes of
/// the bitmap that are not zero.
fn bitmap_from_data(
    data: &[u16],
    bitmap: &mut [u8],
    min_non_zero: &mut u16,
    max_non_zero: &mut u16,
) {
    bitmap[..BITMAP_SIZE].fill(0);

    for &value in data {
        bitmap[(value >> 3) as usize] |= 1 << (value & 7);
    }

    // zero is not explicitly stored in the bitmap; we assume that the data always contain zeroes
    bitmap[0] &= !1;
    *min_non_zero = (BITMAP_SIZE - 1) as u16;
    *max_non_zero = 0;

    for i in 0..BITMAP_SIZE {
        if bitmap[i] != 0 {
            if *min_non_zero as usize > i {
                *min_non_zero = i as u16;
            }
            if (*max_non_zero as usize) < i {
                *max_non_zero = i as u16;
            }
        }
    }
}

/// `forwardLutFromBitmap`: the table from a value to its place among the values that occur.
/// The answer is the largest such place.
fn forward_lut_from_bitmap(bitmap: &[u8], lut: &mut [u16]) -> u16 {
    let mut k = 0i32;

    for i in 0..USHORT_RANGE {
        if i == 0 || (bitmap[i >> 3] & (1 << (i & 7))) != 0 {
            lut[i] = k as u16;
            k += 1;
        } else {
            lut[i] = 0;
        }
    }

    (k - 1) as u16 // maximum value stored in lut[]
}

/// `reverseLutFromBitmap`: the table from a place back to its value.
fn reverse_lut_from_bitmap(bitmap: &[u8], lut: &mut [u16]) -> u16 {
    let mut k = 0usize;

    for i in 0..USHORT_RANGE {
        if i == 0 || (bitmap[i >> 3] & (1 << (i & 7))) != 0 {
            lut[k] = i as u16;
            k += 1;
        }
    }

    let n = k as i32 - 1;

    while k < USHORT_RANGE {
        lut[k] = 0;
        k += 1;
    }

    n as u16 // maximum k where lut[k] is non-zero
}

/// `CompressPiz`: a block of `num_lines` lines of `data_width` pixels, whose channels are
/// `channel_sizes` 16-bit words a pixel each (one for a half, two for a float or an
/// integer), or the block as it is if that is not smaller.
///
/// The original writes into a buffer of 8192 bytes and twice the block, which its caller
/// hands over, and writes past its end for a small block of many different values (the
/// bitmap alone can take 8192 bytes, and the table of code lengths comes after it). Here
/// there is room for the most a block can come to, so such a block comes out as the original
/// would write it with room enough.
pub(crate) fn compress_piz(
    input: &[u8],
    channel_sizes: &[usize],
    data_width: i32,
    num_lines: i32,
) -> Vec<u8> {
    let in_size = input.len();
    let mut bitmap = vec![0u8; BITMAP_SIZE];
    let mut min_non_zero = 0u16;
    let mut max_non_zero = 0u16;

    // Assume `inSize` is multiple of 2 or 4.
    let mut tmp_buffer = vec![0u16; in_size / 2];

    // (start, nx, ny, size) of each channel in `tmp_buffer`.
    let mut channel_data = Vec::with_capacity(channel_sizes.len());
    let mut tmp_buffer_end = 0usize;

    for &size in channel_sizes {
        channel_data.push((tmp_buffer_end, data_width, num_lines, size as i32));
        tmp_buffer_end += (data_width * num_lines) as usize * size;
    }

    {
        let mut ptr = 0usize;
        let mut ends: Vec<usize> = channel_data.iter().map(|cd| cd.0).collect();
        for _y in 0..num_lines {
            for (i, cd) in channel_data.iter().enumerate() {
                let n = (cd.1 * cd.3) as usize;
                for k in 0..n {
                    tmp_buffer[ends[i] + k] =
                        u16::from_le_bytes([input[ptr + 2 * k], input[ptr + 2 * k + 1]]);
                }
                ptr += n * 2;
                ends[i] += n;
            }
        }
    }

    bitmap_from_data(
        &tmp_buffer,
        &mut bitmap,
        &mut min_non_zero,
        &mut max_non_zero,
    );

    let mut lut = vec![0u16; USHORT_RANGE];
    let max_value = forward_lut_from_bitmap(&bitmap, &mut lut);
    // `applyLut`.
    for value in tmp_buffer.iter_mut() {
        *value = lut[*value as usize];
    }

    // Store range compression info in _outBuffer.
    // (The two numbers and the bitmap, the Huffman header and its table of at most 65537
    // lengths of 6 bits each, and at most 17 bits a word.)
    let mut out = vec![0u8; 8192 + 65536 + 3 * in_size + 256];
    let mut buf = 0usize;

    out[buf..buf + 2].copy_from_slice(&min_non_zero.to_le_bytes());
    buf += 2;
    out[buf..buf + 2].copy_from_slice(&max_non_zero.to_le_bytes());
    buf += 2;

    if min_non_zero <= max_non_zero {
        let n = (max_non_zero - min_non_zero) as usize + 1;
        out[buf..buf + n]
            .copy_from_slice(&bitmap[min_non_zero as usize..min_non_zero as usize + n]);
        buf += n;
    }

    // Apply wavelet encoding.
    for cd in &channel_data {
        for j in 0..cd.3 {
            wav2_encode(
                &mut tmp_buffer[cd.0 + j as usize..],
                cd.1,
                cd.3,
                cd.2,
                cd.1 * cd.3,
                max_value,
            );
        }
    }

    // Apply Huffman encoding; append the result to _outBuffer.
    // length header(4byte), then huff data. Initialize length header with zero, then later
    // fill it by `length`.
    let length_ptr = buf;
    out[buf..buf + 4].fill(0);
    buf += 4;

    let length = huf_compress(&tmp_buffer, &mut out[buf..]);
    out[length_ptr..length_ptr + 4].copy_from_slice(&(length as i32).to_le_bytes());

    let out_size = buf + length;

    // Use uncompressed data when compressed data is larger than uncompressed.
    // (Issue 40)
    if out_size >= in_size {
        return input.to_vec();
    }
    out.truncate(out_size);
    out
}

/// `DecompressPiz`: a block of `tmp_buf_size_in_bytes` bytes into `out`, from the `in_len`
/// bytes at the start of `input`, which goes on to the end of the file.
pub(crate) fn decompress_piz(
    out: &mut [u8],
    input: &[u8],
    tmp_buf_size_in_bytes: usize,
    in_len: usize,
    channel_sizes: &[usize],
    data_width: i32,
    num_lines: i32,
) -> bool {
    if in_len == tmp_buf_size_in_bytes {
        // Data is not compressed (Issue 40).
        out[..in_len].copy_from_slice(&input[..in_len]);
        return true;
    }

    let mut bitmap = vec![0u8; BITMAP_SIZE];

    // minimum length for PIZ is 4 bytes
    if in_len < 4 {
        return false;
    }

    let mut read_len = 0usize;
    let mut ptr = 0usize;
    let min_non_zero = u16::from_le_bytes([input[0], input[1]]) as usize;
    let max_non_zero = u16::from_le_bytes([input[2], input[3]]) as usize;
    ptr += 4;
    read_len += 4;

    if max_non_zero >= BITMAP_SIZE {
        return false;
    }

    if min_non_zero <= max_non_zero {
        let n = max_non_zero - min_non_zero + 1;
        if n + read_len > in_len {
            // Input too short
            return false;
        }

        bitmap[min_non_zero..min_non_zero + n].copy_from_slice(&input[ptr..ptr + n]);
        ptr += n;
        read_len += n;
    } else {
        // Issue 194
        if min_non_zero == BITMAP_SIZE - 1 && max_non_zero == 0 {
            // OK. all pixels are zero. And no need to read `bitmap` data.
        } else {
            // invalid minNonZero/maxNonZero combination.
            return false;
        }
    }

    let mut lut = vec![0u16; USHORT_RANGE];
    let max_value = reverse_lut_from_bitmap(&bitmap, &mut lut);

    // Huffman decoding
    if read_len + 4 > in_len {
        return false;
    }

    let length = i32::from_le_bytes([input[ptr], input[ptr + 1], input[ptr + 2], input[ptr + 3]]);
    ptr += 4;

    // (The original compares as an unsigned number, so a sum below zero is too large too.)
    let end = ptr as i64 + length as i64;
    if end < 0 || end as u64 > in_len as u64 {
        return false;
    }

    let mut tmp_buffer = vec![0u16; tmp_buf_size_in_bytes / 2];
    huf_uncompress(&input[ptr..], length as i64, &mut tmp_buffer);

    // Wavelet decoding
    let mut channel_data = Vec::with_capacity(channel_sizes.len());
    let mut tmp_buffer_end = 0usize;

    for &size in channel_sizes {
        channel_data.push((tmp_buffer_end, data_width, num_lines, size as i32));
        tmp_buffer_end += (data_width * num_lines) as usize * size;
    }

    for cd in &channel_data {
        for j in 0..cd.3 {
            wav2_decode(
                &mut tmp_buffer[cd.0 + j as usize..],
                cd.1,
                cd.3,
                cd.2,
                cd.1 * cd.3,
                max_value,
            );
        }
    }

    // Expand the pixel data to their original range.
    for value in tmp_buffer.iter_mut() {
        *value = lut[*value as usize];
    }

    let mut out_ptr = 0usize;
    let mut ends: Vec<usize> = channel_data.iter().map(|cd| cd.0).collect();
    for _y in 0..num_lines {
        for (i, cd) in channel_data.iter().enumerate() {
            let n = (cd.1 * cd.3) as usize;
            for k in 0..n {
                out[out_ptr + 2 * k..out_ptr + 2 * k + 2]
                    .copy_from_slice(&tmp_buffer[ends[i] + k].to_le_bytes());
            }
            out_ptr += n * 2;
            ends[i] += n;
        }
    }

    true
}
