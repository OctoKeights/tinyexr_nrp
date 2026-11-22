//! Throughput of saving and loading a synthetic 1024 by 768 image of half-float A, B, G and
//! R, with each compression.
//!
//! ```sh
//! cargo run --release --example bench
//! ```
//!
//! Each figure is the best of several runs, in megapixels per second. A compression's name,
//! and a number of runs, may be given to measure only that one.
use std::time::Instant;
use tinyexr_nrp::{
    Box2i, ChannelData, ChannelInfo, Compression, ExrHeader, ExrImage, PixelType, load_image,
    parse_header, parse_version, save_image,
};

const WIDTH: i32 = 1024;
const HEIGHT: i32 = 768;

/// A channel of halves: smooth, with a little noise, like a photograph.
fn channel(c: usize) -> ChannelData {
    let mut state = 1u64 + c as u64;
    let values = (0..(WIDTH * HEIGHT) as usize)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 60) as u16;
            let (x, y) = (i % WIDTH as usize, i / WIDTH as usize);
            if c == 0 {
                0x3C00
            } else {
                0x3000 + ((x * (c + 1) + y * (4 - c)) / 3) as u16 % 0x0C00 + noise
            }
        })
        .collect();
    ChannelData::Half(values)
}

fn best(runs: usize, mut run: impl FnMut()) -> f64 {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let only = std::env::args().nth(1);
    let runs = std::env::args()
        .nth(2)
        .map_or(5, |n| n.parse().expect("a number of runs"));
    let names = ["A", "B", "G", "R"];
    let image = ExrImage {
        images: (0..4).map(channel).collect(),
        width: WIDTH,
        height: HEIGHT,
        num_channels: 4,
        ..ExrImage::default()
    };
    let raw = (WIDTH * HEIGHT * 4 * 2) as usize;
    let mpixels = f64::from(WIDTH * HEIGHT) / 1e6;
    for (name, compression_type) in [
        ("none", Compression::NONE),
        ("rle", Compression::RLE),
        ("zips", Compression::ZIPS),
        ("zip", Compression::ZIP),
        ("piz", Compression::PIZ),
        ("pxr24", Compression::PXR24),
        ("b44", Compression::B44),
        #[cfg(feature = "dwa")]
        ("dwaa", Compression::DWAA),
    ] {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        let header = ExrHeader {
            channels: names
                .iter()
                .map(|name| ChannelInfo {
                    name: name.to_string(),
                    pixel_type: PixelType::HALF,
                    x_sampling: 1,
                    y_sampling: 1,
                    p_linear: 0,
                })
                .collect(),
            pixel_types: vec![PixelType::HALF; 4],
            requested_pixel_types: vec![PixelType::HALF; 4],
            compression_type,
            data_window: Box2i {
                min_x: 0,
                min_y: 0,
                max_x: WIDTH - 1,
                max_y: HEIGHT - 1,
            },
            ..ExrHeader::default()
        };
        let file = save_image(&image, &header).expect("an .exr file");
        let save = best(runs, || {
            std::hint::black_box(save_image(&image, &header).unwrap());
        });
        let load = best(runs, || {
            let version = parse_version(&file).unwrap();
            let mut header = parse_header(version, &file).unwrap();
            std::hint::black_box(load_image(&mut header, &file).unwrap());
        });
        println!(
            "{name:6} save {:7.2} Mpixel/s   load {:7.2} Mpixel/s   {:.1}% of {raw}",
            mpixels / save,
            mpixels / load,
            100.0 * file.len() as f64 / raw as f64
        );
    }
}
