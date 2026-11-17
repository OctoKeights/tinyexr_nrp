#!/bin/sh
# Compares tinyexr_nrp with tinyexr (https://github.com/syoyo/tinyexr) v3.4.0.
#
#   MINIZ=<checkout of miniz> ZFP=<checkout of zfp 1.0.1> \
#       compare/compare.sh <checkout of tinyexr> [count] [seeds] [dumps]
#
# compare/dump_tinyexr.cc is built against its tinyexr.h without threads, with the miniz in
# $MINIZ (https://github.com/richgel999/miniz, the one miniz_nrp is the port of), and with ZFP
# compression (TINYEXR_USE_ZFP) against the zfp library v1.0.1 in $ZFP
# (https://github.com/LLNL/zfp), built as its own build has it by default; with memory that
# starts as zeros and with overflow that wraps (-fwrapv), so that the damaged files among those
# it reads have one answer. examples/dump.rs is built from the port; both run the same made-up
# images for each seed and what they print is compared byte for byte. Dumps, each with a tenth
# of [count] (200,000 by default) as the number of images or files: tinyexr_write, made-up
# images saved by both sides; tinyexr_read, images saved by tinyexr, a third of them then
# damaged, and read four ways (the C++ side writes its files to a file, which the Rust side
# reads); tinyexr_zfp_write and tinyexr_zfp_read, the same of images made for ZFP compression;
# tinyexr_parts, files of many parts saved and read; and tinyexr_deep, deep files read. Each is
# of the crate's functions that do exactly as tinyexr does (`..._as_tinyexr`).
set -u
if [ $# -lt 1 ]; then
    echo "usage: $0 <checkout of tinyexr> [count] [seeds] [dumps]" >&2
    exit 2
fi
here=$(cd "$(dirname "$0")/.." && pwd)
source=$1
blocks=${2:-200000}
seeds=${3:-3}
dumps=${4:-tinyexr_write tinyexr_read tinyexr_parts tinyexr_deep tinyexr_zfp_write tinyexr_zfp_read}
tmp=${TMPDIR:-/tmp}/tinyexr_nrp-compare
mkdir -p "$tmp" || exit 2
if ! [ -f "$source/tinyexr.h" ]; then
    echo "$source does not look like checkout of tinyexr" >&2
    exit 2
fi

(cd "$here" && cargo build --release -q --ignore-rust-version --example dump) || exit 2
rust=$(cd "$here" && cargo metadata --format-version 1 --no-deps 2>/dev/null | sed 's/.*"target_directory":"\([^"]*\)".*/\1/')/release/examples/dump

if [ -z "${MINIZ:-}" ] || [ ! -f "$MINIZ/miniz_tdef.c" ]; then
    echo "no miniz: put its checkout in \$MINIZ" >&2
    exit 2
fi
if [ -z "${ZFP:-}" ] || [ ! -f "$ZFP/include/zfp.h" ]; then
    echo "no zfp: put its checkout, v1.0.1, in \$ZFP" >&2
    exit 2
fi
# (The header miniz's build would make, and no .zip code.)
mkdir -p "$tmp/miniz" "$tmp/miniz-obj" "$tmp/zfp-obj" && echo "#define MINIZ_EXPORT" > "$tmp/miniz/miniz_export.h" || exit 2
(cd "$tmp/miniz-obj" && rm -f ./*.o && ${CC:-cc} -O2 -c -DMINIZ_NO_ARCHIVE_APIS -I "$tmp/miniz" -I "$MINIZ" "$MINIZ/miniz.c" "$MINIZ/miniz_tdef.c" "$MINIZ/miniz_tinfl.c") || exit 2
(cd "$tmp/zfp-obj" && rm -f ./*.o && ${CC:-cc} -O2 -fwrapv -ffp-contract=off -c -I "$ZFP/include" "$ZFP"/src/*.c) || exit 2
${CXX:-c++} -O2 -fwrapv -ffp-contract=off -DMINIZ_NO_ARCHIVE_APIS -I "$tmp/miniz" -I "$MINIZ" -I "$source" -I "$ZFP/include" "$here/compare/dump_tinyexr.cc" "$tmp"/miniz-obj/*.o "$tmp"/zfp-obj/*.o -o "$tmp/dump-tinyexr" || exit 2

bad=0
for what in $dumps; do
    extra=
    case $what in
    tinyexr_write | tinyexr_read | tinyexr_parts | tinyexr_deep | tinyexr_zfp_write | tinyexr_zfp_read)
        reference=$tmp/dump-tinyexr
        count=$((blocks / 10))
        [ "$what" = tinyexr_read ] && extra=$tmp/exr-files.bin
        [ "$what" = tinyexr_zfp_read ] && extra=$tmp/exr-zfp.bin
        [ "$what" = tinyexr_parts ] && extra=$tmp/exr-parts.bin
        [ "$what" = tinyexr_deep ] && extra=$tmp/exr-deep.bin
        ;;
    *)
        echo "unknown dump $what" >&2
        exit 2
        ;;
    esac
    seed=1
    while [ "$seed" -le "$seeds" ]; do
        # shellcheck disable=SC2086
        "$reference" "$what" "$seed" "$count" $extra > "$tmp/reference.txt" 2> /dev/null
        # shellcheck disable=SC2086
        "$rust" "$what" "$seed" "$count" $extra > "$tmp/rust.txt"
        if [ ! -s "$tmp/reference.txt" ]; then
            echo "$what, seed $seed: NOT COMPARED, the original's side printed nothing"
            bad=$((bad + 1))
        elif cmp -s "$tmp/reference.txt" "$tmp/rust.txt"; then
            echo "$what, seed $seed: the same ($(tail -1 "$tmp/rust.txt"))"
        else
            echo "$what, seed $seed: DIFFERENT"
            bad=$((bad + 1))
        fi
        seed=$((seed + 1))
    done
done
[ "$bad" -eq 0 ]
