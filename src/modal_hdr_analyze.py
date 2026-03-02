"""Modal-based HDR analysis pipeline.

Offloads hdr_analyzer_mvp to Modal.com cloud instances for faster processing.
The binary is compiled inside the container image (x86_64 Ubuntu 24.04) to match
the FFmpeg library ABI, mirroring the CI release workflow exactly.

The apt FFmpeg packages don't include hevc_cuvid, so the binary falls back to
software decoding automatically. On L4's fast x86_64 CPU, software decode of
4K HEVC runs at ~40-60 fps — faster than ARM64 even without GPU decode.

Usage:
    modal run src/modal_hdr_analyze.py --input video.mkv
    modal run src/modal_hdr_analyze.py --input video.mkv --sample-rate 3
    modal run src/modal_hdr_analyze.py --input video.mkv --downscale 2
    modal run src/modal_hdr_analyze.py --input video.mkv --output custom.measurements.bin
"""

import modal
import subprocess
import time
import uuid
from pathlib import Path

# hdr-analyze workspace root, resolved relative to this script
_HDR_ANALYZE_ROOT = (Path(__file__).parent.parent.parent / "hdr-analyze").resolve()

app = modal.App("modal-hdr-analyze")

# ---------------------------------------------------------------------------
# Persistent volume for streamed file transfer (shared with modal_ffmpeg)
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("ffmpeg-working", create_if_missing=True)
VOL_MOUNT = "/vol"

# ---------------------------------------------------------------------------
# Analyze image — CUDA runtime + FFmpeg dev libs + compiled hdr_analyzer_mvp
#
# Build mirrors the CI release workflow (release.yml lines 150-158):
#   - Ubuntu 24.04 x86_64
#   - apt FFmpeg dev libraries
#   - BINDGEN_EXTRA_CLANG_ARGS=-I/usr/include/x86_64-linux-gnu
#   - cargo build --release --package hdr_analyzer_mvp
# ---------------------------------------------------------------------------
analyze_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "pkg-config",
        "clang",
        "lld",
        "libclang-dev",
        "curl",
        "libavformat-dev",
        "libavcodec-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libavdevice-dev",
        "libswscale-dev",
    )
    # Install Rust stable toolchain (minimal profile — no docs/clippy)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
    )
    # Copy workspace source files (surgical — excludes target/, .git/, tests)
    .add_local_file(
        _HDR_ANALYZE_ROOT / "Cargo.toml",
        "/build/Cargo.toml",
        copy=True,
    )
    .add_local_file(
        _HDR_ANALYZE_ROOT / "Cargo.lock",
        "/build/Cargo.lock",
        copy=True,
    )
    .add_local_dir(
        _HDR_ANALYZE_ROOT / ".cargo",
        "/build/.cargo",
        copy=True,
    )
    .add_local_dir(
        _HDR_ANALYZE_ROOT / "hdr_analyzer_mvp",
        "/build/hdr_analyzer_mvp",
        copy=True,
    )
    .add_local_dir(
        _HDR_ANALYZE_ROOT / "mkvdolby",
        "/build/mkvdolby",
        copy=True,
    )
    .add_local_dir(
        _HDR_ANALYZE_ROOT / "verifier",
        "/build/verifier",
        copy=True,
    )
    # Compile inside the container, then discard source tree
    .run_commands(
        "cd /build && "
        "BINDGEN_EXTRA_CLANG_ARGS='-I/usr/include/x86_64-linux-gnu' "
        "/root/.cargo/bin/cargo build --release --package hdr_analyzer_mvp",
        "cp /build/target/release/hdr_analyzer_mvp /usr/local/bin/",
        "rm -rf /build",
        "ls -la /usr/local/bin/hdr_analyzer_mvp",
        "hdr_analyzer_mvp --version",
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,video,utility"})
)


# ---------------------------------------------------------------------------
# Remote analysis function
# ---------------------------------------------------------------------------


@app.function(
    image=analyze_image,
    gpu="L4",
    cpu=8,
    memory=8192,
    timeout=7200,
    volumes={VOL_MOUNT: volume},
)
def analyze_hdr(
    job_id: str,
    input_name: str,
    output_name: str,
    sample_rate: int = 1,
    scene_threshold: float = 0.3,
    downscale: int = 1,
) -> dict:
    """Analyze an HDR video file and produce a madVR measurement .bin file.

    Reads input from and writes output to the shared Volume.
    """
    input_path = f"{VOL_MOUNT}/jobs/{job_id}/input/{input_name}"
    output_dir = f"{VOL_MOUNT}/jobs/{job_id}/output"
    output_path = f"{output_dir}/{output_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "hdr_analyzer_mvp",
        input_path,
        "-o",
        output_path,
        "--sample-rate",
        str(sample_rate),
        "--scene-threshold",
        str(scene_threshold),
        "--downscale",
        str(downscale),
    ]

    print(f"[cmd] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # Forward binary output so Modal logs capture progress/errors
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        tail = result.stderr[-1500:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"hdr_analyzer_mvp failed (rc={result.returncode}):\n{tail}"
        )

    output_size = Path(output_path).stat().st_size
    print(f"[time] Analysis: {elapsed:.2f}s")
    print(f"[size] output: {output_size} bytes")

    volume.commit()
    return {"analysis_time": elapsed, "output_size": output_size}


# ---------------------------------------------------------------------------
# Volume cleanup helper
# ---------------------------------------------------------------------------


def _cleanup_job(job_id: str):
    """Remove job directory from volume. Non-fatal on failure."""
    try:
        volume.remove_file(f"jobs/{job_id}", recursive=True)
    except Exception as e:
        print(f"[warn] Volume cleanup failed for job {job_id}: {e}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    input: str,
    output: str | None = None,
    sample_rate: int = 1,
    scene_threshold: float = 0.3,
    downscale: int = 1,
):
    """Upload a video file to Modal, run HDR analysis, download the .bin result.

    Options:
        --input           Path to local HDR video file (required)
        --output          Output .bin path (default: {stem}.measurements.bin)
        --sample-rate     Analyze every Nth frame (default: 1; use 3 for ~3x speedup)
        --scene-threshold Scene cut threshold (default: 0.3)
        --downscale       Analysis resolution divisor (default: 1; use 2 for ~2x speedup)
    """
    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output is None:
        output = f"{input_path.stem}.measurements.bin"
    output_path = Path(output)

    input_mb = input_path.stat().st_size / (1024 * 1024)
    job_id = uuid.uuid4().hex[:12]

    print("=" * 55)
    print(f"  Job:          {job_id}")
    print(f"  Input:        {input_path} ({input_mb:.2f} MB)")
    print(f"  Output:       {output_path}")
    print(f"  Sample rate:  every {sample_rate} frame(s)")
    print(f"  Scene thresh: {scene_threshold}")
    print(f"  Downscale:    {downscale}x")
    print("=" * 55)

    input_name = input_path.name
    output_name = output_path.name
    vol_input_path = f"jobs/{job_id}/input/{input_name}"
    vol_output_path = f"jobs/{job_id}/output/{output_name}"

    # --- Upload phase ---
    print(f"\n[1/4] Uploading {input_mb:.2f} MB to volume...")
    t_upload = time.time()
    try:
        with volume.batch_upload() as batch:
            batch.put_file(input_path, vol_input_path)
        t_upload = time.time() - t_upload
        print(f"[1/4] Upload complete ({t_upload:.2f}s)")
    except Exception:
        _cleanup_job(job_id)
        raise

    # --- Analyze phase ---
    print("\n[2/4] Analyzing HDR on Modal (L4, 8 CPU)...")
    t_analyze_start = time.time()
    try:
        result = analyze_hdr.remote(
            job_id=job_id,
            input_name=input_name,
            output_name=output_name,
            sample_rate=sample_rate,
            scene_threshold=scene_threshold,
            downscale=downscale,
        )
        t_analyze_wall = time.time() - t_analyze_start
        print(
            f"[2/4] Analysis complete ({t_analyze_wall:.2f}s wall, {result['analysis_time']:.2f}s binary)"
        )
    except Exception:
        _cleanup_job(job_id)
        raise

    # --- Download phase ---
    output_size = result["output_size"]
    print(f"\n[3/4] Downloading {output_size} bytes from volume...")
    t_download = time.time()
    try:
        with open(output_path, "wb") as f:
            for chunk in volume.read_file(vol_output_path):
                f.write(chunk)
        t_download = time.time() - t_download
        print(f"[3/4] Download complete ({t_download:.2f}s)")
    except Exception:
        if output_path.exists():
            output_path.unlink()
        _cleanup_job(job_id)
        raise

    # --- Cleanup phase ---
    print("\n[4/4] Cleaning up volume...")
    _cleanup_job(job_id)
    print("[4/4] Cleanup complete")

    # Summary
    t_total = t_upload + t_analyze_wall + t_download
    print()
    print("=" * 55)
    print(f"  Input:       {input_mb:.2f} MB")
    print(f"  Output:      {output_size} bytes ({output_size / 1024:.1f} KB)")
    print(f"  Upload:      {t_upload:.2f}s")
    print(
        f"  Analyze:     {t_analyze_wall:.2f}s (wall) / {result['analysis_time']:.2f}s (binary)"
    )
    print(f"  Download:    {t_download:.2f}s")
    print(f"  Total:       {t_total:.2f}s")
    print("=" * 55)
    print(f"\nDone: {output_path}")
