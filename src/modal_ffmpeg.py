"""
Modal-based FFmpeg video processing pipeline.

Offloads FFmpeg encoding to Modal.com cloud instances when local resources
are insufficient. Supports three encoding workflows:

  hevc      — GPU-accelerated HEVC encoding via NVENC (L4 GPU)
  x265      — CPU-based libx265 software encoding (no GPU required, higher quality)
  composite — BL+EL NLQ compositing piped to NVENC on L4 GPU (Dolby Vision FEL)

All workflows use FFmpeg 8.x (BtbN static builds). Files are streamed via
Modal Volumes — no full-file byte transfer over RPC. Each job gets an isolated
directory on the volume, cleaned up after completion.

Usage:
    # GPU HEVC encoding (fast, hardware-accelerated)
    modal run src/modal_ffmpeg.py --mode hevc --input sample.mkv

    # CPU x265 encoding (slower, software encoder, excellent quality)
    modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv

    # High-performance x265 (16 CPU cores instead of default 8)
    modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv --perf high

    # BL+EL compositing + NVENC (Dolby Vision FEL pipeline)
    modal run src/modal_ffmpeg.py --mode composite \\
        --bl BL.hevc --el EL.hevc --rpu RPU.bin \\
        --width 3840 --height 2160 --output composited.mkv

    # With options
    modal run src/modal_ffmpeg.py --mode hevc --input sample.mkv --preset p5 --qp 20
    modal run src/modal_ffmpeg.py --mode x265 --input sample.mkv --preset slow --crf 18
"""

import modal
import subprocess
import time
import uuid
from pathlib import Path

# Resolve the mkvdolby binary relative to this script file, so it works
# regardless of the working directory from which `modal run` is invoked.
_MKVDOLBY_BIN = (Path(__file__).parent.parent / "bin" / "mkvdolby").resolve()

app = modal.App("modal-ffmpeg")

# ---------------------------------------------------------------------------
# Persistent volume for streamed file transfer (free tier, no RPC size limits)
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("ffmpeg-working", create_if_missing=True)
VOL_MOUNT = "/vol"

# ---------------------------------------------------------------------------
# BtbN FFmpeg 8.x static build (GPL, includes libx265 + NVENC support)
# ---------------------------------------------------------------------------
FFMPEG_BUILD_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

# GPU image — NVIDIA CUDA runtime + FFmpeg with NVENC
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("wget", "xz-utils")
    .run_commands(
        f"wget -q {FFMPEG_BUILD_URL}",
        "tar xf ffmpeg-master-latest-linux64-gpl.tar.xz",
        "cp ffmpeg-master-latest-linux64-gpl/bin/ff* /usr/local/bin/",
        "rm -rf ffmpeg-master-latest-linux64-gpl*",
        "ffmpeg -version | head -1",
        "ffmpeg -encoders 2>/dev/null | grep hevc_nvenc",
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,video,utility"})
)

# CPU image — lightweight Debian + FFmpeg (for x265 software encoding)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "xz-utils")
    .run_commands(
        f"wget -q {FFMPEG_BUILD_URL}",
        "tar xf ffmpeg-master-latest-linux64-gpl.tar.xz",
        "cp ffmpeg-master-latest-linux64-gpl/bin/ff* /usr/local/bin/",
        "rm -rf ffmpeg-master-latest-linux64-gpl*",
        "ffmpeg -version | head -1",
        "ffmpeg -encoders 2>/dev/null | grep libx265",
    )
)

# GPU composite image — CUDA + FFmpeg + mkvdolby (for BL+EL compositing + NVENC)
gpu_composite_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("wget", "xz-utils")
    .run_commands(
        f"wget -q {FFMPEG_BUILD_URL}",
        "tar xf ffmpeg-master-latest-linux64-gpl.tar.xz",
        "cp ffmpeg-master-latest-linux64-gpl/bin/ff* /usr/local/bin/",
        "rm -rf ffmpeg-master-latest-linux64-gpl*",
        "ffmpeg -version | head -1",
        "ffmpeg -encoders 2>/dev/null | grep hevc_nvenc",
    )
    .add_local_file(_MKVDOLBY_BIN, "/usr/local/bin/mkvdolby", copy=True)
    .run_commands("chmod +x /usr/local/bin/mkvdolby")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,video,utility"})
)


# ---------------------------------------------------------------------------
# GPU workflow — HEVC encoding via NVENC
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=4,
    memory=4096,
    timeout=3600,
    volumes={VOL_MOUNT: volume},
)
def encode_hevc(
    job_id: str,
    input_name: str,
    output_name: str,
    preset: str = "p5",
    qp: int = 18,
    scale: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Encode video to HEVC using NVENC hardware encoder on an L4 GPU.

    Reads input from and writes output to the shared Volume.
    """
    # Verify NVENC availability
    check = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    if "hevc_nvenc" not in check.stdout:
        raise RuntimeError("hevc_nvenc not available — GPU/driver issue")
    print("[ok] hevc_nvenc encoder available")

    input_path = f"{VOL_MOUNT}/jobs/{job_id}/input/{input_name}"
    output_dir = f"{VOL_MOUNT}/jobs/{job_id}/output"
    output_path = f"{output_dir}/{output_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", input_path]

    if scale:
        w, h = scale.split("x")
        cmd.extend(["-vf", f"scale={w}:{h}:flags=lanczos"])

    cmd.extend(
        [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            preset,
            "-tune",
            "hq",
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-profile:v",
            "main10",
            "-pix_fmt",
            "p010le",
            "-c:a",
            "copy",
        ]
    )

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(output_path)

    print(f"[cmd] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        tail = result.stderr[-1500:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"FFmpeg failed (rc={result.returncode}):\n{tail}")

    output_size = Path(output_path).stat().st_size
    print(f"[time] GPU encode: {elapsed:.2f}s")
    print(f"[size] output: {output_size / (1024 * 1024):.2f} MB")

    volume.commit()
    return {"encode_time": elapsed, "output_size": output_size}


# ---------------------------------------------------------------------------
# GPU composite workflow — BL+EL NLQ compositing piped to NVENC
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_composite_image,
    gpu="L4",
    cpu=8,
    memory=8192,
    timeout=7200,
    volumes={VOL_MOUNT: volume},
)
def composite_and_encode(
    job_id: str,
    bl_name: str,
    el_name: str,
    rpu_name: str,
    output_name: str,
    width: int,
    height: int,
    fps_num: int = 24000,
    fps_den: int = 1001,
    preset: str = "p5",
    qp: int = 18,
    extra_args: list[str] | None = None,
) -> dict:
    """Composite BL+EL via NLQ and encode to HEVC using NVENC on an L4 GPU.

    Pipes mkvdolby composite-pipe directly into ffmpeg NVENC — no intermediate file.
    Reads BL/EL/RPU from and writes output to the shared Volume.
    """
    # Verify NVENC availability
    check = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    if "hevc_nvenc" not in check.stdout:
        raise RuntimeError("hevc_nvenc not available — GPU/driver issue")
    print("[ok] hevc_nvenc encoder available")

    # Verify mkvdolby binary
    check_mkv = subprocess.run(
        ["mkvdolby", "--version"], capture_output=True, text=True
    )
    if check_mkv.returncode != 0:
        raise RuntimeError("mkvdolby binary not available in container")
    print(f"[ok] mkvdolby: {check_mkv.stdout.strip()}")

    job_dir = f"{VOL_MOUNT}/jobs/{job_id}"
    input_dir = f"{job_dir}/input"
    output_dir = f"{job_dir}/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bl_path = f"{input_dir}/{bl_name}"
    el_path = f"{input_dir}/{el_name}"
    rpu_path = f"{input_dir}/{rpu_name}"
    output_path = f"{output_dir}/{output_name}"

    # Build composite command (stdout = raw yuv420p16le frames)
    composite_cmd = [
        "mkvdolby",
        "composite-pipe",
        "--bl",
        bl_path,
        "--el",
        el_path,
        "--rpu",
        rpu_path,
        "-w",
        str(width),
        "-H",
        str(height),
        "--fps-num",
        str(fps_num),
        "--fps-den",
        str(fps_den),
    ]

    # Build encode command (stdin = raw yuv420p16le frames)
    framerate = f"{fps_num}/{fps_den}"
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p16le",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        framerate,
        "-i",
        "pipe:0",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        preset,
        "-tune",
        "hq",
        "-rc",
        "constqp",
        "-qp",
        str(qp),
        "-profile:v",
        "main10",
        "-pix_fmt",
        "p010le",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        "-color_range",
        "tv",
    ]

    if extra_args:
        encode_cmd.extend(extra_args)

    encode_cmd.append(output_path)

    print(f"[composite] {' '.join(composite_cmd)}")
    print(f"[encode]    {' '.join(encode_cmd)}")

    # Pipe composite → encode
    t0 = time.time()
    composite_proc = subprocess.Popen(
        composite_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encode_proc = subprocess.Popen(
        encode_cmd,
        stdin=composite_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Allow composite_proc to receive SIGPIPE if encode_proc exits
    composite_proc.stdout.close()

    encode_stdout, encode_stderr = encode_proc.communicate()
    composite_proc.wait()
    elapsed = time.time() - t0

    # Check composite exit
    if composite_proc.returncode != 0:
        comp_err = (
            composite_proc.stderr.read().decode() if composite_proc.stderr else ""
        )
        raise RuntimeError(
            f"mkvdolby composite-pipe failed (rc={composite_proc.returncode}):\n{comp_err[-1500:]}"
        )

    # Check encode exit
    if encode_proc.returncode != 0:
        enc_err = encode_stderr.decode() if encode_stderr else "(no stderr)"
        raise RuntimeError(
            f"FFmpeg encode failed (rc={encode_proc.returncode}):\n{enc_err[-1500:]}"
        )

    output_size = Path(output_path).stat().st_size
    print(f"[time] Composite + encode: {elapsed:.2f}s")
    print(f"[size] output: {output_size / (1024 * 1024):.2f} MB")

    volume.commit()
    return {"encode_time": elapsed, "output_size": output_size}


# ---------------------------------------------------------------------------
# CPU workflow — x265 software encoding
# ---------------------------------------------------------------------------

# Shared x265 encoding logic used by both default and high-perf functions.
# The pools parameter MUST match the CPU reservation to prevent x265 from
# spawning extra threads beyond reservation — Modal bills actual CPU usage,
# so unrestricted threads cause over-billing.


def _x265_encode(
    input_path: str,
    output_path: str,
    preset: str,
    crf: int,
    pools: int,
    scale: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Encode video to HEVC using libx265 software encoder (CPU).

    Reads/writes directly on Volume mount paths.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract HDR10 params marker from extra_args if present
    hdr_params: str | None = None
    filtered_extra: list[str] = []
    if extra_args:
        it = iter(extra_args)
        for arg in it:
            if arg == "--x265-hdr-params":
                hdr_params = next(it, None)
            else:
                filtered_extra.append(arg)

    cmd = ["ffmpeg", "-y", "-i", input_path]

    if scale:
        w, h = scale.split("x")
        cmd.extend(["-vf", f"scale={w}:{h}:flags=lanczos"])

    # Build x265-params: always include pools, merge HDR10 params if provided
    x265_params = f"pools={pools}"
    if hdr_params:
        x265_params = f"{hdr_params}:{x265_params}"

    cmd.extend(
        [
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-x265-params",
            x265_params,
            "-tag:v",
            "hvc1",
            "-c:a",
            "copy",
        ]
    )

    if filtered_extra:
        cmd.extend(filtered_extra)

    cmd.append(output_path)

    print(f"[cmd] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        tail = result.stderr[-1500:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"FFmpeg failed (rc={result.returncode}):\n{tail}")

    output_size = Path(output_path).stat().st_size
    print(f"[time] CPU encode: {elapsed:.2f}s")
    print(f"[size] output: {output_size / (1024 * 1024):.2f} MB")

    return {"encode_time": elapsed, "output_size": output_size}


# Default x265: 8 CPU / 4 GB — best cost/performance balance
@app.function(
    image=cpu_image,
    cpu=8,
    memory=4096,
    timeout=7200,
    volumes={VOL_MOUNT: volume},
)
def encode_x265(
    job_id: str,
    input_name: str,
    output_name: str,
    preset: str = "medium",
    crf: int = 18,
    scale: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Encode video to HEVC using libx265 (8 CPU cores, default).

    Reads input from and writes output to the shared Volume.
    """
    input_path = f"{VOL_MOUNT}/jobs/{job_id}/input/{input_name}"
    output_path = f"{VOL_MOUNT}/jobs/{job_id}/output/{output_name}"
    result = _x265_encode(
        input_path,
        output_path,
        preset,
        crf,
        pools=8,
        scale=scale,
        extra_args=extra_args,
    )
    volume.commit()
    return result


# High-performance x265: 16 CPU / 4 GB — ~1.7x faster, ~18% higher cost
@app.function(
    image=cpu_image,
    cpu=16,
    memory=4096,
    timeout=7200,
    volumes={VOL_MOUNT: volume},
)
def encode_x265_fast(
    job_id: str,
    input_name: str,
    output_name: str,
    preset: str = "medium",
    crf: int = 18,
    scale: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Encode video to HEVC using libx265 (16 CPU cores, high-performance).

    Same quality as encode_x265 but ~1.7x faster at ~18% higher cost.
    Use --perf high on CLI or call encode_x265_fast.remote() directly.
    """
    input_path = f"{VOL_MOUNT}/jobs/{job_id}/input/{input_name}"
    output_path = f"{VOL_MOUNT}/jobs/{job_id}/output/{output_name}"
    result = _x265_encode(
        input_path,
        output_path,
        preset,
        crf,
        pools=16,
        scale=scale,
        extra_args=extra_args,
    )
    volume.commit()
    return result


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
# Composite mode helper
# ---------------------------------------------------------------------------


def _run_composite(
    bl: str | None,
    el: str | None,
    rpu: str | None,
    output: str | None,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    preset: str | None,
    qp: int,
    master_display: str | None,
    max_cll: str | None,
):
    """Upload BL+EL+RPU to Volume, run composite_and_encode on Modal, download result."""
    if not bl or not el or not rpu:
        raise ValueError("Composite mode requires --bl, --el, and --rpu paths.")

    bl_path = Path(bl)
    el_path = Path(el)
    rpu_path = Path(rpu)

    for p, label in [(bl_path, "BL"), (el_path, "EL"), (rpu_path, "RPU")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} file not found: {p}")

    if output is None:
        output = f"{bl_path.stem}_composited.mkv"
    output_path = Path(output)

    # Build extra_args for VUI color tags (HDR10 signaling via NVENC)
    extra_args: list[str] | None = None
    if master_display or max_cll:
        extra_args = [
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
        ]

    total_mb = sum(
        p.stat().st_size / (1024 * 1024) for p in [bl_path, el_path, rpu_path]
    )
    job_id = uuid.uuid4().hex[:12]

    print("=" * 55)
    print(f"  Job:    {job_id}")
    print("  Mode:   COMPOSITE (BL+EL → NVENC)")
    print(f"  BL:     {bl_path} ({bl_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"  EL:     {el_path} ({el_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"  RPU:    {rpu_path} ({rpu_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"  Res:    {width}x{height} @ {fps_num}/{fps_den}")
    p = preset or "p5"
    print(f"  Codec:  hevc_nvenc  preset={p}  qp={qp}")
    print("=" * 55)

    bl_name = bl_path.name
    el_name = el_path.name
    rpu_name = rpu_path.name
    output_name = output_path.name
    vol_output_path = f"jobs/{job_id}/output/{output_name}"

    # --- Upload phase ---
    print(f"\n[1/4] Uploading {total_mb:.2f} MB (BL+EL+RPU) to volume...")
    t_upload = time.time()
    try:
        with volume.batch_upload() as batch:
            batch.put_file(bl_path, f"jobs/{job_id}/input/{bl_name}")
            batch.put_file(el_path, f"jobs/{job_id}/input/{el_name}")
            batch.put_file(rpu_path, f"jobs/{job_id}/input/{rpu_name}")
        t_upload = time.time() - t_upload
        print(f"[1/4] Upload complete ({t_upload:.2f}s)")
    except Exception:
        _cleanup_job(job_id)
        raise

    # --- Composite + encode phase ---
    print("\n[2/4] Compositing + encoding on Modal (L4 GPU)...")
    t_encode_start = time.time()
    try:
        result = composite_and_encode.remote(
            job_id=job_id,
            bl_name=bl_name,
            el_name=el_name,
            rpu_name=rpu_name,
            output_name=output_name,
            width=width,
            height=height,
            fps_num=fps_num,
            fps_den=fps_den,
            preset=preset or "p5",
            qp=qp,
            extra_args=extra_args,
        )
        t_encode_wall = time.time() - t_encode_start
        print(
            f"[2/4] Composite+encode complete ({t_encode_wall:.2f}s wall, {result['encode_time']:.2f}s pipeline)"
        )
    except Exception:
        _cleanup_job(job_id)
        raise

    # --- Download phase ---
    output_mb = result["output_size"] / (1024 * 1024)
    print(f"\n[3/4] Downloading {output_mb:.2f} MB from volume...")
    t_download = time.time()
    try:
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in volume.read_file(vol_output_path):
                f.write(chunk)
                downloaded += len(chunk)
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

    # Results
    t_total = t_upload + t_encode_wall + t_download
    print()
    print("=" * 55)
    print(f"  Upload (BL+EL+RPU): {total_mb:.2f} MB in {t_upload:.2f}s")
    print(f"  Output:     {output_mb:.2f} MB")
    print(
        f"  Encode:     {t_encode_wall:.2f}s (wall) / {result['encode_time']:.2f}s (pipeline)"
    )
    print(f"  Download:   {t_download:.2f}s")
    print(f"  Total:      {t_total:.2f}s")
    print("=" * 55)
    print(f"\nDone: {output_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    input: str = "sample.mkv",
    output: str | None = None,
    mode: str = "hevc",
    perf: str = "default",
    # HEVC (GPU) params
    preset: str | None = None,
    qp: int = 18,
    # x265 (CPU) params
    crf: int = 18,
    # HDR10 metadata (x265 only)
    master_display: str | None = None,
    max_cll: str | None = None,
    # Shared
    scale: str | None = None,
    # Composite mode params
    bl: str | None = None,
    el: str | None = None,
    rpu: str | None = None,
    width: int = 3840,
    height: int = 2160,
    fps_num: int = 24000,
    fps_den: int = 1001,
):
    """Send a local video file to Modal for FFmpeg encoding.

    Modes:
        hevc      — GPU NVENC hardware encoder (fast, default)
        x265      — CPU libx265 software encoder (slower, high quality)
        composite — BL+EL NLQ compositing + NVENC (Dolby Vision FEL)

    Performance (x265 only):
        default — 8 CPU cores, best cost/performance balance
        high    — 16 CPU cores, ~1.7x faster at ~18% higher cost

    HDR10 metadata (x265 only):
        --master-display  — SMPTE ST 2086 mastering display string
                            e.g. "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,50)"
        --max-cll         — MaxCLL,MaxFALL e.g. "1000,400"
    """
    if mode not in ("hevc", "x265", "composite"):
        raise ValueError(
            f"Unknown mode '{mode}'. Choose 'hevc', 'x265', or 'composite'."
        )
    if perf not in ("default", "high"):
        raise ValueError(f"Unknown perf '{perf}'. Choose 'default' or 'high'.")

    # Dispatch composite mode separately
    if mode == "composite":
        _run_composite(
            bl=bl,
            el=el,
            rpu=rpu,
            output=output,
            width=width,
            height=height,
            fps_num=fps_num,
            fps_den=fps_den,
            preset=preset,
            qp=qp,
            master_display=master_display,
            max_cll=max_cll,
        )
        return

    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Default output name
    if output is None:
        output = f"{input_path.stem}_{mode}{input_path.suffix}"
    output_path = Path(output)

    # Build HDR10 extra_args when metadata flags are provided
    extra_args: list[str] | None = None
    if master_display or max_cll:
        if mode == "hevc":
            # NVENC: set VUI color tags. The DV RPU controls display mapping,
            # so mastering display SEI is not critical for playback.
            extra_args = [
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
                "-colorspace",
                "bt2020nc",
                "-color_range",
                "tv",
            ]
        else:
            # x265: use x265-params for HDR10 signaling
            extra_args = ["-pix_fmt", "yuv420p10le", "-profile:v", "main10"]
            hdr_parts = [
                "colorprim=bt2020",
                "transfer=smpte2084",
                "colormatrix=bt2020nc",
                "hdr-opt=1",
                "repeat-headers=1",
            ]
            if master_display:
                hdr_parts.append(f"master-display={master_display}")
            if max_cll:
                hdr_parts.append(f"max-cll={max_cll}")
            extra_args.append("--x265-hdr-params")
            extra_args.append(":".join(hdr_parts))

    input_mb = input_path.stat().st_size / (1024 * 1024)
    job_id = uuid.uuid4().hex[:12]

    # Print job summary
    print("=" * 55)
    print(f"  Job:    {job_id}")
    print(f"  Mode:   {mode.upper()}")
    if mode == "x265":
        cpu_label = "16 CPU (high-perf)" if perf == "high" else "8 CPU (default)"
        print(f"  Perf:   {cpu_label}")
    print(f"  Input:  {input_path} ({input_mb:.2f} MB)")
    if scale:
        print(f"  Scale:  {scale}")
    if mode == "hevc":
        p = preset or "p5"
        print(f"  Codec:  hevc_nvenc  preset={p}  qp={qp}")
    else:
        p = preset or "medium"
        print(f"  Codec:  libx265  preset={p}  crf={crf}")
    if master_display:
        print(f"  HDR10:  master-display={master_display}")
    if max_cll:
        print(f"  HDR10:  max-cll={max_cll}")
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

    # --- Encode phase ---
    print(f"\n[2/4] Encoding ({mode.upper()})...")
    t_encode_start = time.time()
    try:
        if mode == "hevc":
            result = encode_hevc.remote(
                job_id=job_id,
                input_name=input_name,
                output_name=output_name,
                preset=preset or "p5",
                qp=qp,
                scale=scale,
                extra_args=extra_args,
            )
        elif perf == "high":
            result = encode_x265_fast.remote(
                job_id=job_id,
                input_name=input_name,
                output_name=output_name,
                preset=preset or "medium",
                crf=crf,
                scale=scale,
                extra_args=extra_args,
            )
        else:
            result = encode_x265.remote(
                job_id=job_id,
                input_name=input_name,
                output_name=output_name,
                preset=preset or "medium",
                crf=crf,
                scale=scale,
                extra_args=extra_args,
            )
        t_encode_wall = time.time() - t_encode_start
        print(
            f"[2/4] Encode complete ({t_encode_wall:.2f}s wall, {result['encode_time']:.2f}s encode)"
        )
    except Exception:
        _cleanup_job(job_id)
        raise

    # --- Download phase ---
    output_mb = result["output_size"] / (1024 * 1024)
    print(f"\n[3/4] Downloading {output_mb:.2f} MB from volume...")
    t_download = time.time()
    try:
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in volume.read_file(vol_output_path):
                f.write(chunk)
                downloaded += len(chunk)
        t_download = time.time() - t_download
        print(f"[3/4] Download complete ({t_download:.2f}s)")
    except Exception:
        # Remove partial local file
        if output_path.exists():
            output_path.unlink()
        _cleanup_job(job_id)
        raise

    # --- Cleanup phase ---
    print("\n[4/4] Cleaning up volume...")
    _cleanup_job(job_id)
    print("[4/4] Cleanup complete")

    # Results
    t_total = t_upload + t_encode_wall + t_download
    print()
    print("=" * 55)
    print(f"  Input:      {input_mb:.2f} MB")
    print(f"  Output:     {output_mb:.2f} MB")
    if input_mb > 0:
        print(f"  Ratio:      {output_mb / input_mb:.2f}x")
    print(f"  Upload:     {t_upload:.2f}s")
    print(
        f"  Encode:     {t_encode_wall:.2f}s (wall) / {result['encode_time']:.2f}s (FFmpeg)"
    )
    print(f"  Download:   {t_download:.2f}s")
    print(f"  Total:      {t_total:.2f}s")
    print("=" * 55)
    print(f"\nDone: {output_path}")
