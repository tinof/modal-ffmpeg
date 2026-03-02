"""Integration tests for modal_hdr_analyze.py.

Tests run the actual Modal pipeline end-to-end and validate output correctness
using the local verifier binary. They are automatically skipped when Modal
credentials are not configured or required fixtures are missing.

Requirements:
    - Modal configured: `modal token set` or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET
    - Test sample:  /home/ubuntu/modal-ffmpeg/test_hdr10_sample.mkv  (329 KB)
    - Verifier:     /home/ubuntu/hdr-analyze/target/release/verifier

Run all:
    cd /home/ubuntu/modal-ffmpeg && pytest tests/test_hdr_analyze.py -v

Run a specific test:
    pytest tests/test_hdr_analyze.py::test_sample_nvdec -v -s
"""

import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_MODAL_APP = _REPO_ROOT / "src" / "modal_hdr_analyze.py"
_SAMPLE_MKV = _REPO_ROOT / "test_hdr10_sample.mkv"
_VERIFIER = Path("/home/ubuntu/hdr-analyze/target/release/verifier")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _modal_auth_ok() -> bool:
    """Return True when Modal credentials are configured and usable."""
    r = subprocess.run(["modal", "token", "list"], capture_output=True, text=True)
    return r.returncode == 0


def _verify_bin(bin_path: Path) -> subprocess.CompletedProcess:
    """Run the local verifier against a .bin file and return the result."""
    return subprocess.run(
        [str(_VERIFIER), str(bin_path)],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Markers / skip conditions
# ---------------------------------------------------------------------------

_need_modal = pytest.mark.skipif(
    not _modal_auth_ok(), reason="Modal credentials not configured"
)
_need_sample = pytest.mark.skipif(
    not _SAMPLE_MKV.exists(), reason=f"Test sample not found: {_SAMPLE_MKV}"
)
_need_verifier = pytest.mark.skipif(
    not _VERIFIER.exists(), reason=f"Verifier binary not found: {_VERIFIER}"
)


# ---------------------------------------------------------------------------
# Local tests (no Modal required)
# ---------------------------------------------------------------------------


def test_help_exits_cleanly():
    """--help must exit 0 and list all expected flags."""
    r = subprocess.run(
        ["modal", "run", str(_MODAL_APP), "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"--help exited non-zero:\n{r.stderr}"
    for flag in ("--input", "--output", "--sample-rate", "--downscale", "--hwaccel"):
        assert flag in r.stdout, f"Expected flag {flag!r} missing from --help output"


def test_invalid_hwaccel_rejected(tmp_path):
    """An invalid --hwaccel value must cause a non-zero exit."""
    r = subprocess.run(
        [
            "modal",
            "run",
            str(_MODAL_APP),
            "--input",
            str(_SAMPLE_MKV) if _SAMPLE_MKV.exists() else "/dev/null",
            "--hwaccel",
            "bogus",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode != 0, (
        "Expected non-zero exit for invalid --hwaccel value, got 0.\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}"
    )


# ---------------------------------------------------------------------------
# Integration tests (Modal required)
# ---------------------------------------------------------------------------


@_need_modal
@_need_sample
@_need_verifier
def test_sample_software_decode(tmp_path):
    """Pipeline with --hwaccel none: upload → software decode → download → verify."""
    output = tmp_path / "sample_sw.measurements.bin"
    r = subprocess.run(
        [
            "modal",
            "run",
            str(_MODAL_APP),
            "--input",
            str(_SAMPLE_MKV),
            "--output",
            str(output),
            "--hwaccel",
            "none",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"modal run failed:\n{r.stdout}\n{r.stderr}"
    assert output.exists(), "Output .bin was not downloaded"
    assert output.stat().st_size > 1024, "Output .bin is suspiciously small"

    # Structural validation via local verifier
    v = _verify_bin(output)
    assert v.returncode == 0, f"Verifier failed:\n{v.stdout}"
    assert "All data integrity checks passed" in v.stdout

    # Confirm software decode was reported
    assert "HW Decode:   software" in r.stdout, (
        "Expected software decode label in output summary"
    )


@_need_modal
@_need_sample
@_need_verifier
def test_sample_nvdec(tmp_path):
    """Pipeline with --hwaccel auto: NVDEC attempted; valid output regardless of fallback.

    NVDEC may or may not be available on Modal's L4 containers at runtime.
    The test asserts:
      - The pipeline completes successfully either way
      - The output passes verifier validation
      - The summary clearly reports which decode path was used
    """
    output = tmp_path / "sample_nvdec.measurements.bin"
    r = subprocess.run(
        [
            "modal",
            "run",
            str(_MODAL_APP),
            "--input",
            str(_SAMPLE_MKV),
            "--output",
            str(output),
            "--hwaccel",
            "auto",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"modal run failed:\n{r.stdout}\n{r.stderr}"
    assert output.exists(), "Output .bin was not downloaded"
    assert output.stat().st_size > 1024, "Output .bin is suspiciously small"

    # Structural validation via local verifier
    v = _verify_bin(output)
    assert v.returncode == 0, f"Verifier failed:\n{v.stdout}"
    assert "All data integrity checks passed" in v.stdout

    # Confirm the summary reports a known decode path (not "unknown")
    assert "HW Decode:" in r.stdout, "Expected HW Decode line in summary"
    assert "unknown" not in r.stdout, "hwaccel_used should not be 'unknown'"

    # Report which path was taken (informational — not a failure condition)
    if "HW Decode:   nvdec" in r.stdout:
        print("\n[info] NVDEC hardware decode was used")
    else:
        print("\n[info] Software decode was used (NVDEC fell back or unavailable)")


@_need_modal
@_need_sample
@_need_verifier
def test_nvdec_and_software_outputs_are_equivalent(tmp_path):
    """NVDEC and software outputs must have the same frame/scene counts.

    Frame-level PQ values may differ slightly due to decoder rounding, but
    the file structure (frame count, scene count) must match exactly.
    """
    out_sw = tmp_path / "sw.measurements.bin"
    out_nv = tmp_path / "nv.measurements.bin"

    for hwaccel, out in (("none", out_sw), ("auto", out_nv)):
        r = subprocess.run(
            [
                "modal",
                "run",
                str(_MODAL_APP),
                "--input",
                str(_SAMPLE_MKV),
                "--output",
                str(out),
                "--hwaccel",
                hwaccel,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, f"modal run ({hwaccel}) failed:\n{r.stdout}\n{r.stderr}"

    def _extract(label: str, text: str) -> str:
        for line in text.splitlines():
            if label in line:
                return line.strip()
        return ""

    sw_verify = _verify_bin(out_sw).stdout
    nv_verify = _verify_bin(out_nv).stdout

    sw_frames = _extract("Frame count:", sw_verify)
    nv_frames = _extract("Frame count:", nv_verify)
    sw_scenes = _extract("Scene count:", sw_verify)
    nv_scenes = _extract("Scene count:", nv_verify)

    assert sw_frames == nv_frames, (
        f"Frame counts differ between software and NVDEC:\n  SW: {sw_frames}\n  NV: {nv_frames}"
    )
    assert sw_scenes == nv_scenes, (
        f"Scene counts differ between software and NVDEC:\n  SW: {sw_scenes}\n  NV: {nv_scenes}"
    )


@_need_modal
@_need_sample
def test_auto_output_naming(tmp_path):
    """Without --output, the file is named {stem}.measurements.bin in the invocation cwd.

    Note: Modal executes the local entrypoint in-process, so it inherits the cwd
    from the subprocess launched here via the cwd= argument.
    """
    r = subprocess.run(
        [
            "modal",
            "run",
            str(_MODAL_APP),
            "--input",
            str(_SAMPLE_MKV),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=300,
    )
    assert r.returncode == 0, f"modal run failed:\n{r.stdout}\n{r.stderr}"

    # The auto-named output must be {stem}.measurements.bin in cwd (tmp_path)
    expected = tmp_path / "test_hdr10_sample.measurements.bin"
    assert expected.exists(), (
        f"Expected auto-named output at {expected}.\n"
        "If this fails, Modal may have changed how local entrypoints inherit cwd."
    )
    assert expected.stat().st_size > 1024
