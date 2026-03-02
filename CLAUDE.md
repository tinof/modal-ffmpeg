# modal-ffmpeg

FFmpeg video encoding pipeline on Modal.com. Offloads encoding to cloud GPU/CPU when local resources aren't sufficient.

## Architecture

Single Modal app (`src/modal_ffmpeg.py`) with four functions:
- `encode_hevc` — GPU function (L4, hevc_nvenc hardware encoder, preset p5)
- `encode_x265` — CPU function (8 cores, libx265 software encoder, default)
- `encode_x265_fast` — CPU function (16 cores, libx265, high-performance)
- `composite_and_encode` — GPU function (L4, BL+EL NLQ compositing piped to NVENC)

Three container images:
- `gpu_image` — CUDA + FFmpeg (for `encode_hevc`)
- `cpu_image` — Debian + FFmpeg (for `encode_x265` / `encode_x265_fast`)
- `gpu_composite_image` — CUDA + FFmpeg + mkvdolby binary (for `composite_and_encode`)

All use BtbN FFmpeg 8.x static builds. Files stream through a Modal Volume (`ffmpeg-working`) —
no full-file byte transfer over RPC. Each job gets an isolated directory (`jobs/{uuid}/`) on the
volume, cleaned up after completion.

**Standard flow:** Local upload → Volume → Remote encode → Volume → Local download → Cleanup

**Composite flow:** Upload BL+EL+RPU → Volume → `mkvdolby composite-pipe | ffmpeg NVENC` on L4 → Volume → Download → Cleanup

## Key files

- `src/modal_ffmpeg.py` — the entire application (container images, encoding functions, CLI entrypoint)
- `src/benchmark_x265.py` — benchmarking script for CPU/memory config optimization
- `bin/mkvdolby` — cross-compiled x86_64 binary for NLQ compositing (gitignored, built from `hdr-analyze/mkvdolby`)

## Running

```bash
# GPU HEVC encoding
modal run src/modal_ffmpeg.py --mode hevc --input sample.mkv

# CPU x265 encoding
modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv
modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv --perf high

# BL+EL compositing + NVENC (Dolby Vision FEL pipeline)
modal run src/modal_ffmpeg.py --mode composite \
  --bl BL.hevc --el EL.hevc --rpu RPU.bin \
  --width 3840 --height 2160 --output composited.mkv
```

## Performance tiers (x265)

| Tier | Flag | CPU | Memory | Cost | Speed |
|------|------|-----|--------|------|-------|
| Default | `--perf default` | 8 cores | 4 GB | ~$0.0024/encode* | Baseline |
| High | `--perf high` | 16 cores | 4 GB | ~$0.0028/encode* | ~1.7x faster |

*Per 11s 1080p clip. Scales linearly with content duration.

For AI agents: use `--perf high` when faster turnaround is needed, or call `encode_x265_fast.remote()` directly.

## Cost optimization notes

- Modal Volumes are free — no storage cost for the transient job files.
- x265 thread pools are limited via `-x265-params pools=N` to match the CPU reservation.
  Without this, x265 auto-detects host cores (Modal soft limit = reservation + 16) and
  spawns threads beyond the reservation. Modal bills MAX(reservation, actual usage), so
  unrestricted threads cause silent over-billing.
- Memory is set to 4 GB (x265) / 8 GB (composite). Benchmarked peak usage: ~1 GB (1080p), ~2.8 GB (4K).
- CPU cost dominates memory cost by ~6:1 per unit, so CPU core count is the main cost lever.
- Modal pricing: $0.0000131/core/sec CPU, $0.00000222/GiB/sec memory.
- L4 GPU ($0.80/hr) is faster than T4 ($0.59/hr) for NVENC; shorter wall time compensates.
- p5 preset is ~2x faster than p7 with <1% quality difference at constant QP.

## Test files

- `sample.mkv` — 1080p, 7.9 MB
- `sample_4k.mkv` — 4K, 18 MB
- `test_hdr10_sample.mkv` — HDR10, 329 KB

## Cross-compiling mkvdolby

The host is ARM64 (Oracle), Modal containers are x86_64. Use `cargo-zigbuild`:

```bash
cd /home/ubuntu/hdr-analyze
cargo zigbuild --release --package mkvdolby --target x86_64-unknown-linux-gnu
cp target/x86_64-unknown-linux-gnu/release/mkvdolby /home/ubuntu/modal-ffmpeg/bin/
```

The `.cargo/config.toml` has `[target.x86_64-unknown-linux-gnu] rustflags = []` to avoid passing ARM-specific flags.

## Conventions

- Output files are auto-named `*_hevc.mkv`, `*_x265.mkv`, or `*_composited.mkv` (gitignored)
- All source code lives in `src/`
- `bin/` contains cross-compiled binaries (gitignored)
- No unnecessary files — keep the repo minimal
- Container images are built with BtbN FFmpeg builds (GPL, includes libx265 + NVENC)
- GPU images use `nvidia/cuda:12.8.1-runtime-ubuntu24.04`
- CPU image uses `modal.Image.debian_slim(python_version="3.12")`
- Volume `ffmpeg-working` is used for streamed file transfer; job dirs are cleaned up automatically
- Timeouts: GPU encode 3600s (1hr), CPU/composite 7200s (2hr) — sized for large REMUX files
