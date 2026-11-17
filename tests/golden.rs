//! The port of tinyexr against results recorded from the original, so that a change shows
//! without a C++ compiler at hand.
//!
//! `tests/data/tinyexr_write_expected.txt` is what `compare/dump_tinyexr.cc` printed for
//! `tinyexr_write_small 5 300` (see `compare/compare.sh`): a hash of the files tinyexr saves
//! of made-up images, in scanlines and in tiles, with every compression it writes, or of its
//! message where it saves none.
//!
//! `tests/data/exr_files.bin` holds the files that program made with tinyexr for
//! `tinyexr_read_small 11 100`: such images saved, a third of them damaged.
//! `tests/data/tinyexr_read_expected.txt` is what it printed for them: a hash of what tinyexr
//! reads of each, or of the message it refuses it with.
//!
//! `tests/data/tinyexr_parts_expected.txt` is what it printed for `tinyexr_parts 7 60`: files
//! of many parts saved, and read; `tests/data/exr_parts.bin` the files it read (a third of
//! them damaged) with what came of each. `tests/data/tinyexr_deep_expected.txt` and
//! `tests/data/exr_deep.bin` are the same for `tinyexr_deep 13 80`, of made-up deep files.
//!
//! `tests/data/tinyexr_zfp_write_expected.txt` is what it printed for `tinyexr_zfp_write 5 300`,
//! and `tests/data/tinyexr_zfp_read_expected.txt` for `tinyexr_zfp_read 11 100` (whose files
//! are `tests/data/exr_zfp.bin`): the same of images for ZFP compression, of floats with the
//! attributes of its settings, built with `TINYEXR_USE_ZFP` and zfp 1.0.1.

mod made_up;

#[test]
fn writes_as_tinyexr_does() {
    // (And reads each file back: what was stored as it was comes back as it was.)
    let printed = made_up::tinyexr_write("tinyexr_write_small", 5, 300);
    let expected = include_str!("data/tinyexr_write_expected.txt");
    for (line, (ours, theirs)) in printed.lines().zip(expected.lines()).enumerate() {
        assert_eq!(ours, theirs, "line {}", line + 1);
    }
    assert_eq!(printed.lines().count(), expected.lines().count());
}

#[test]
fn reads_as_tinyexr_does() {
    let printed = made_up::tinyexr_read(
        "tinyexr_read_small",
        100,
        include_bytes!("data/exr_files.bin"),
    );
    let expected = include_str!("data/tinyexr_read_expected.txt");
    for (line, (ours, theirs)) in printed.lines().zip(expected.lines()).enumerate() {
        assert_eq!(ours, theirs, "line {}", line + 1);
    }
    assert_eq!(printed.lines().count(), expected.lines().count());
}

#[test]
fn zfp_is_written_and_read_as_tinyexr_does() {
    let printed = made_up::tinyexr_write("tinyexr_zfp_write", 5, 300);
    assert_eq!(printed, include_str!("data/tinyexr_zfp_write_expected.txt"));
    let printed =
        made_up::tinyexr_read("tinyexr_zfp_read", 100, include_bytes!("data/exr_zfp.bin"));
    assert_eq!(printed, include_str!("data/tinyexr_zfp_read_expected.txt"));
}

#[test]
fn files_of_many_parts_are_written_and_read_as_tinyexr_does() {
    let printed =
        made_up::tinyexr_parts("tinyexr_parts", 7, 60, include_bytes!("data/exr_parts.bin"));
    assert_eq!(printed, include_str!("data/tinyexr_parts_expected.txt"));
}

#[test]
fn deep_files_are_read_as_tinyexr_does() {
    let printed = made_up::tinyexr_deep("tinyexr_deep", 80, include_bytes!("data/exr_deep.bin"));
    assert_eq!(printed, include_str!("data/tinyexr_deep_expected.txt"));
}

/// Files of many parts and deep files damaged far worse than the comparison's: whatever is in
/// them, the readers give an answer and do not panic.
#[test]
fn badly_damaged_parts_and_deep_files_are_read_or_refused() {
    use tinyexr_nrp as tinyexr;
    let mut r = made_up::Random { state: 5 };
    let (mut read, mut refused) = (0, 0);
    for (file, with_wish) in [
        (&include_bytes!("data/exr_parts.bin")[..], true),
        (&include_bytes!("data/exr_deep.bin")[..], false),
    ] {
        for _ in 0..6 {
            let mut at = 0;
            while at < file.len() {
                let number = |at: &mut usize| {
                    *at += 4;
                    u32::from_le_bytes(file[*at - 4..*at].try_into().unwrap()) as usize
                };
                if with_wish {
                    number(&mut at);
                }
                let _flags = number(&mut at);
                let size = number(&mut at);
                let mut bytes = file[at..at + size].to_vec();
                at += size;
                if bytes.is_empty() {
                    continue;
                }
                for _ in 0..1 + r.next() % 4 {
                    let within = if r.next().is_multiple_of(2) {
                        bytes.len().min(500)
                    } else {
                        bytes.len()
                    };
                    let place = r.next() as usize % within;
                    bytes[place] = if r.next().is_multiple_of(4) {
                        0
                    } else {
                        r.next() as u8
                    };
                }
                if r.next().is_multiple_of(8) {
                    bytes.truncate(r.next() as usize % bytes.len());
                }
                if !with_wish {
                    match tinyexr::load_deep_exr_from_memory(&bytes) {
                        Ok(_) => read += 1,
                        Err(_) => refused += 1,
                    }
                    continue;
                }
                let Ok(version) = tinyexr::parse_exr_version_from_memory(&bytes) else {
                    refused += 1;
                    continue;
                };
                let Ok(mut headers) =
                    tinyexr::parse_exr_multipart_header_from_memory(&version, &bytes)
                else {
                    refused += 1;
                    continue;
                };
                // (Not the sizes that ask for more memory than a test should take.)
                if headers.iter().any(made_up::tinyexr_left) {
                    continue;
                }
                match tinyexr::load_exr_multipart_image_from_memory(&mut headers, &bytes) {
                    Ok(_) => read += 1,
                    Err(_) => refused += 1,
                }
            }
        }
    }
    assert!(read > 0 && refused > 0);
}

/// Files damaged far worse than the comparison's: whatever is in them, the reader gives an
/// image or a message, and does not panic. (`ROUNDS` over the recorded files; more by hand
/// when the reader is changed.)
#[test]
fn badly_damaged_files_are_read_or_refused() {
    use tinyexr_nrp as tinyexr;
    const ROUNDS: u32 = 12;
    let file = include_bytes!("data/exr_files.bin");
    let mut r = made_up::Random { state: 3 };
    let (mut read, mut refused) = (0, 0);
    for _ in 0..ROUNDS {
        let mut at = 0;
        while at < file.len() {
            let number = |at: &mut usize| {
                *at += 4;
                u32::from_le_bytes(file[*at - 4..*at].try_into().unwrap()) as usize
            };
            let _wish = number(&mut at);
            let size = number(&mut at);
            let mut bytes = file[at..at + size].to_vec();
            at += size;
            if bytes.is_empty() {
                continue;
            }
            // A few bytes made up, more often near the start, where the header and the table
            // of offsets are; now and then the end cut off.
            for _ in 0..1 + r.next() % 4 {
                let within = if r.next().is_multiple_of(2) {
                    bytes.len().min(500)
                } else {
                    bytes.len()
                };
                let place = r.next() as usize % within;
                bytes[place] = if r.next().is_multiple_of(4) {
                    0
                } else {
                    r.next() as u8
                };
            }
            if r.next().is_multiple_of(8) {
                bytes.truncate(r.next() as usize % bytes.len());
            }

            let Ok(version) = tinyexr::parse_exr_version_from_memory(&bytes) else {
                refused += 1;
                continue;
            };
            let Ok(mut header) = tinyexr::parse_exr_header_from_memory(&version, &bytes) else {
                refused += 1;
                continue;
            };
            // (Not the sizes that ask for more memory than a test should take.)
            if made_up::tinyexr_left(&header) {
                continue;
            }
            // (Tinyexr's way and the format's, by turns.)
            let loaded = if r.next().is_multiple_of(2) {
                tinyexr::load_exr_image_from_memory(&mut header, &bytes)
            } else {
                tinyexr::load_exr_image_from_memory_as_tinyexr(&mut header, &bytes)
            };
            match loaded {
                Ok(_) => read += 1,
                Err(_) => refused += 1,
            }
            let _ = tinyexr::load_exr_from_memory(&bytes);
            let _ = tinyexr::load_exr_from_memory_as_tinyexr(&bytes);
        }
    }
    assert!(read > 0 && refused > 0);
}
