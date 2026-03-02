# modal-ffmpeg

Offload FFmpeg video encoding to [Modal.com](https://modal.com) cloud instances when local resources aren't enough. Supports GPU-accelerated HEVC encoding via NVENC and CPU-based x265 software encoding, both using FFmpeg 8.x.

## Workflows

| Mode | Codec | Runs on | Best for |
|------|-------|---------|----------|
| `hevc` | `hevc_nvenc` | T4 GPU | Fast hardware encoding |
| `x265` | `libx265` | CPU (8 cores default, 16 high-perf) | High-quality software encoding |

Both workflows:
- Use FFmpeg 8.x (BtbN static builds from master)
- Stream files via Modal Volume (`ffmpeg-working`) — no file size limit, low memory usage
- Each job is isolated by UUID and cleaned up after completion
- Output is written locally, nothing persists on Modal

## Usage

```bash
# GPU HEVC (fast, hardware-accelerated) — default mode
modal run src/modal_ffmpeg.py --mode hevc --input sample.mkv

# CPU x265 (slower, excellent quality)
modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv

# High-performance x265 (16 CPU cores, ~1.7x faster)
modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv --perf high
```

### HEVC options

```bash
modal run src/modal_ffmpeg.py --mode hevc --input video.mkv \
    --preset p5 \       # p1 (fastest) to p7 (best quality, default)
    --qp 20 \           # constant QP, lower = better quality (default: 18)
    --scale 3840x2160   # optional output resolution
```

### x265 options

```bash
modal run src/modal_ffmpeg.py --mode x265 --input video.mkv \
    --preset slow \     # ultrafast/superfast/.../slow/slower/veryslow (default: medium)
    --crf 22 \          # constant rate factor, lower = better quality (default: 18)
    --perf high \       # 16 CPU cores (default: 8)
    --scale 1920x1080   # optional output resolution
```

### Output naming

Output files are auto-named based on mode: `input_hevc.mkv` or `input_x265.mkv`. Override with `--output`:

```bash
modal run src/modal_ffmpeg.py --mode hevc --input sample.mkv --output my_output.mkv
```

## Benchmarks

### x265 CPU scaling (slow preset, CRF 18, 1080p sample.mkv)

| CPU | Encode time | Wall time | Cost/encode | Scaling efficiency |
|-----|-------------|-----------|-------------|-------------------|
| 2 cores | 74.9s | ~82s | $0.0026 | baseline |
| 4 cores | 33.8s | ~41s | $0.0021 | 111% (super-linear) |
| **8 cores** | **17.7s** | **24.2s** | **$0.0024** | **91%** |
| 16 cores | 14.6s | 21.6s | $0.0028 | 72% |

### x265 CPU scaling (slow preset, CRF 18, 4K sample_4k.mkv)

| CPU | Encode time | Wall time | Cost/encode | Scaling efficiency |
|-----|-------------|-----------|-------------|-------------------|
| 2 cores | 257.5s | ~265s | $0.0090 | baseline |
| 4 cores | 143.2s | ~151s | $0.0088 | 90% |
| **8 cores** | **75.0s** | **~83s** | **$0.0085** | **86%** |
| 16 cores | 44.6s | ~53s | $0.0097 | 72% |

### Local vs Modal (x265 slow preset, including file transfer)

| Where | 1080p wall time | 4K wall time |
|-------|----------------|-------------|
| Local (4 ARM cores) | 58.5s | 246.3s |
| Modal 8 CPU (default) | 24.2s (2.4x faster) | ~83s (3.0x faster) |
| Modal 16 CPU (high) | 21.6s (2.7x faster) | ~53s (4.6x faster) |

Wall times include fixed overhead (Modal app init + Volume upload/download).
Upload and download are streamed, so overhead scales with file size but remains
small relative to encode time for large files.

### Memory impact (8 CPU, slow preset)

Memory has minimal effect on encode speed. Benchmarked peak usage:
- 1080p: ~1 GB peak
- 4K: ~2.8 GB peak

4 GB reservation covers all tested resolutions with headroom.

### Resource configuration

| Function | CPU | Memory | Cost basis |
|----------|-----|--------|------------|
| `encode_hevc` | 4 cores + T4 GPU | 4 GB | GPU: $0.59/hr + CPU/mem |
| `encode_x265` (default) | 8 cores | 4 GB | $0.047/core/hr + $0.008/GiB/hr |
| `encode_x265_fast` (high) | 16 cores | 4 GB | Same rates, more cores |

Thread pools are explicitly limited via `-x265-params pools=N` to match the CPU
reservation. Without this, x265 spawns threads beyond the reservation and Modal
bills for actual usage, causing silent over-billing.

Modal Volumes are free — no storage cost for transient job files.

## Prerequisites

- Python 3.12+
- [Modal](https://modal.com) account and CLI configured (`modal token set`)
- Modal Python SDK: `pip install modal`

## Project structure

```
src/modal_ffmpeg.py     # Modal app — encoding workflows + CLI entrypoint
src/benchmark_x265.py   # CPU/memory benchmarking script
sample.mkv              # Test file (1080p)
sample_4k.mkv           # Test file (4K)
test_hdr10_sample.mkv   # Test file (HDR10)
```

Volume `ffmpeg-working` is created automatically on first run (`create_if_missing=True`).
