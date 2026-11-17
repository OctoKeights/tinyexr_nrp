//! The Rust half of the comparison with tinyexr: runs the port over made-up data and
//! prints a running hash of what comes out, as `compare/dump_tinyexr.cc` does with the original.
//! `compare/compare.sh` compares the two.
//!
//!   dump tinyexr_write <seed> <count>               made-up images saved as .exr files, and
//!   dump tinyexr_write_small <seed> <count>         every fourth the short way as well
//!   dump tinyexr_read <seed> <count> <file>         the .exr files that program wrote to the
//!   dump tinyexr_read_small <seed> <count> <file>   file, each read four ways
//!   dump tinyexr_zfp_write <seed> <count>           the same of images for ZFP compression,
//!   dump tinyexr_zfp_read <seed> <count> <file>     of floats with its settings
//!   dump tinyexr_parts <seed> <count> <file>        files of many parts saved, and the ones
//!                                                   that program wrote to the file read
//!   dump tinyexr_deep <seed> <count> <file>         the deep files that program wrote, read

#[path = "../tests/made_up/mod.rs"]
mod made_up;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(what), Some(seed), Some(count)) = (
        args.get(1),
        args.get(2).and_then(|s| s.parse::<u64>().ok()),
        args.get(3).and_then(|s| s.parse::<u32>().ok()),
    ) else {
        eprintln!("usage: dump <what> <seed> <count>   (see the top of dump.rs)");
        std::process::exit(2);
    };
    let _ = seed;
    match what.as_str() {
        "tinyexr_write" | "tinyexr_write_small" | "tinyexr_zfp_write" => {
            print!("{}", made_up::tinyexr_write(what, seed, count))
        }
        "tinyexr_read" | "tinyexr_read_small" | "tinyexr_zfp_read" | "tinyexr_parts"
        | "tinyexr_deep" => {
            let Some(file) = args.get(4).and_then(|path| std::fs::read(path).ok()) else {
                eprintln!(
                    "usage: dump {what} <seed> <count> <file of files from compare/dump_tinyexr.cc>"
                );
                std::process::exit(2);
            };
            match what.as_str() {
                "tinyexr_parts" => print!("{}", made_up::tinyexr_parts(what, seed, count, &file)),
                "tinyexr_deep" => print!("{}", made_up::tinyexr_deep(what, count, &file)),
                _ => print!("{}", made_up::tinyexr_read(what, count, &file)),
            }
        }
        _ => {
            eprintln!("unknown dump {what}");
            std::process::exit(2);
        }
    }
}
