from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


FINAL_RECEIPT_HOLD_SECONDS = 15.0
TRANSITION_SECONDS = 0.45
WIDTH = 1280
HEIGHT = 720
FPS = 30


def run(command: list[str], *, cwd: Path, allow_fail: bool = False) -> subprocess.CompletedProcess:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode and not allow_fail:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )
    return completed


def ffprobe_duration(path: Path, *, cwd: Path) -> float:
    completed = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        cwd=cwd,
    )
    return float(completed.stdout.strip())


def find_edge() -> Path:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    resolved = shutil.which("msedge")
    if resolved:
        return Path(resolved)
    raise RuntimeError("Microsoft Edge was not found for headless panel export")


def powershell_json(command: str, *, cwd: Path):
    completed = run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=cwd,
        allow_fail=True,
    )
    if completed.returncode:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return completed.stdout.strip()


def system_specs(cwd: Path) -> dict:
    cpu = powershell_json(
        "Get-CimInstance Win32_Processor | "
        "Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json",
        cwd=cwd,
    )
    memory = powershell_json(
        "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json",
        cwd=cwd,
    )
    gpu = powershell_json(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json",
        cwd=cwd,
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "logical_cpu_count": os.cpu_count(),
    }


def first_existing(root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def proof_clip_library(root: Path) -> list[dict]:
    clips = [
        {
            "id": "river_fog_regen",
            "label": "Regenerated river/fog proof clip",
            "path": first_existing(
                root,
                [
                    "outputs/edge_of_the_world_audio_river_smoke/**/edge_audio_river_smoke_5s_letterbox_full_audio.mp4",
                    "outputs/edge_of_the_world_audio_river/**/edge_of_the_world_audio_river_full_v2_letterbox_full_audio.mp4",
                ],
            ),
        },
        {
            "id": "framegen_616",
            "label": "6-1-6 forecast/frame generation proof clip",
            "path": first_existing(
                root,
                ["outputs/flame_walk_forecast/**/child_to_flame_walk_616_forecast_preview_full_audio.mp4"],
            ),
        },
        {
            "id": "raw_render_lane",
            "label": "Native generated render lane proof clip",
            "path": first_existing(
                root,
                [
                    "outputs/weird_occlusion_rs/fade_away_memory_cathedral_full_qsv_rs/fade_away_memory_cathedral_full_qsv_rs.mp4",
                    "outputs/weird_occlusion_rs/glitch_444_house_center_warp_laserfield_full_beast_qsv_rs/glitch_444_house_center_warp_laserfield_full_beast_qsv_rs.mp4",
                ],
            ),
        },
    ]
    return [clip for clip in clips if clip["path"] is not None]


def export_panel(edge: Path, html_path: Path, out_png: Path, slide: int, profile: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    url = html_path.resolve().as_uri() + f"?slide={slide}"
    run(
        [
            str(edge),
            "--headless=new",
            "--disable-background-networking",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            f"--user-data-dir={profile}",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--screenshot={out_png}",
            url,
        ],
        cwd=html_path.parent,
    )


def write_final_receipt_panel(
    out_dir: Path,
    audio_path: Path,
    duration: float,
    clip_library: list[dict],
    specs: dict,
) -> Path:
    gpu_items = specs.get("gpu") or []
    if isinstance(gpu_items, dict):
        gpu_items = [gpu_items]
    gpu_names = ", ".join(str(item.get("Name", "unknown")) for item in gpu_items[:3]) or "detected by manifest"
    cpu = specs.get("cpu") or {}
    memory = specs.get("memory") or {}
    ram_gb = 0.0
    if isinstance(memory, dict):
        ram_gb = float(memory.get("TotalPhysicalMemory", 0)) / (1024**3)
    clip_rows = "\n".join(
        f"<div><b>{clip['label']}</b><span>{clip['path'].name}</span></div>" for clip in clip_library
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  width: 1280px;
  height: 720px;
  overflow: hidden;
  background:
    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px) 0 0 / 40px 40px,
    linear-gradient(0deg, rgba(255,255,255,.04) 1px, transparent 1px) 0 0 / 40px 40px,
    #07101d;
  color: #edf4ff;
  font-family: "Bahnschrift", "Segoe UI", Arial, sans-serif;
}}
.frame {{ position:absolute; inset:34px 42px; border:5px solid #d8e7ff; background:#0d1728; box-shadow:0 30px 80px rgba(0,0,0,.5); }}
.top {{ height:64px; background:#17243d; display:flex; align-items:center; justify-content:space-between; padding:0 30px; font-weight:900; font-size:25px; }}
.body {{ padding:20px 32px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.metric {{ border:3px solid #627da8; background:#111f36; min-height:92px; padding:11px 15px; }}
.metric b {{ display:block; color:#f0c66a; font-size:28px; line-height:1; }}
.metric span {{ display:block; margin-top:8px; color:#aebfde; font-size:14px; font-weight:700; }}
.section {{ margin-top:13px; border:3px solid #2d4366; background:#0a1322; padding:10px 18px; }}
.section h2 {{ margin:0 0 7px; font-size:19px; color:#d8e7ff; }}
.clips {{ display:grid; grid-template-columns:1fr; gap:8px; }}
.clips div {{ display:flex; justify-content:space-between; gap:14px; border-bottom:1px solid #263a59; padding:4px 0; color:#d8e7ff; font-size:13px; }}
.clips span {{ color:#8297bb; text-align:right; }}
.hold {{ position:absolute; right:28px; bottom:14px; color:#f0c66a; font-size:16px; font-weight:900; }}
.seal {{ position:absolute; left:32px; bottom:14px; color:#7f93b5; font-size:14px; font-weight:700; }}
</style>
</head>
<body>
<div class="frame">
  <div class="top"><span>TRUEVISION PROOF REEL RECEIPT</span><span>FINAL 15 SECOND HOLD</span></div>
  <div class="body">
    <div class="grid">
      <div class="metric"><b>{duration:.2f}s</b><span>source audio duration</span></div>
      <div class="metric"><b>1280x720</b><span>presentation output surface</span></div>
      <div class="metric"><b>{FPS} FPS</b><span>timeline cadence</span></div>
      <div class="metric"><b>15.0s</b><span>receipt hold duration</span></div>
      <div class="metric"><b>{len(clip_library)}</b><span>proof clips referenced</span></div>
      <div class="metric"><b>12</b><span>gothic-industrial panels exported</span></div>
      <div class="metric"><b>QSV</b><span>GPU encoder requested</span></div>
      <div class="metric"><b>HIDDEN</b><span>internal backbone not published</span></div>
    </div>
    <div class="section">
      <h2>Machine surface used for this render</h2>
      <div class="clips">
        <div><b>CPU</b><span>{cpu.get('Name', 'detected')} | {cpu.get('NumberOfCores', os.cpu_count())} cores | {cpu.get('NumberOfLogicalProcessors', os.cpu_count())} logical</span></div>
        <div><b>RAM</b><span>{ram_gb:.1f} GiB detected</span></div>
        <div><b>GPU</b><span>{gpu_names}</span></div>
        <div><b>Encoder</b><span>FFmpeg h264_qsv requested, fallback policy available</span></div>
      </div>
    </div>
    <div class="section">
      <h2>Visible receipts shown, private backbone withheld</h2>
      <div class="clips">{clip_rows}</div>
    </div>
  </div>
</div>
</body>
</html>"""
    path = out_dir / "final_receipt_hold.html"
    path.write_text(html, encoding="utf-8")
    return path


def still_segment(
    image: Path,
    segment: Path,
    seconds: float,
    *,
    cwd: Path,
    encoder: str,
) -> str:
    frames = max(1, int(round(seconds * FPS)))
    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image),
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        encoder,
        "-b:v",
        "18M",
        str(segment),
    ]
    result = run(command, cwd=cwd, allow_fail=True)
    if result.returncode == 0:
        return encoder
    fallback = "libx264"
    run(command[:-4] + [fallback, "-crf", "16", str(segment)], cwd=cwd)
    return fallback


def clip_segment(
    source: Path,
    segment: Path,
    seconds: float,
    *,
    cwd: Path,
    encoder: str,
) -> str:
    command = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(source),
        "-t",
        f"{seconds:.3f}",
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-an",
        "-r",
        str(FPS),
        "-c:v",
        encoder,
        "-b:v",
        "18M",
        str(segment),
    ]
    result = run(command, cwd=cwd, allow_fail=True)
    if result.returncode == 0:
        return encoder
    fallback = "libx264"
    run(command[:-4] + [fallback, "-crf", "16", str(segment)], cwd=cwd)
    return fallback


def concat_segments(segments: list[Path], visual_only: Path, *, cwd: Path) -> None:
    list_path = visual_only.parent / "concat_segments.txt"
    lines = [f"file '{segment.as_posix()}'" for segment in segments]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(visual_only)],
        cwd=cwd,
    )


def crossfade_segments(
    segments: list[Path],
    durations: list[float],
    visual_only: Path,
    *,
    cwd: Path,
    encoder: str,
) -> str:
    if len(segments) < 2:
        shutil.copyfile(segments[0], visual_only)
        return "copy"

    inputs: list[str] = []
    for segment in segments:
        inputs.extend(["-i", str(segment)])

    chains = [f"[{index}:v]format=yuv420p,settb=AVTB[v{index}]" for index in range(len(segments))]
    current_label = "v0"
    current_duration = durations[0]
    for index in range(1, len(segments)):
        next_label = f"x{index}"
        offset = max(0.0, current_duration - TRANSITION_SECONDS)
        chains.append(
            f"[{current_label}][v{index}]"
            f"xfade=transition=fade:duration={TRANSITION_SECONDS:.3f}:offset={offset:.3f}"
            f"[{next_label}]"
        )
        current_label = next_label
        current_duration = current_duration + durations[index] - TRANSITION_SECONDS

    filter_complex = ";".join(chains)
    command = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{current_label}]",
        "-an",
        "-r",
        str(FPS),
        "-c:v",
        encoder,
        "-b:v",
        "18M",
        str(visual_only),
    ]
    result = run(command, cwd=cwd, allow_fail=True)
    if result.returncode == 0:
        return encoder

    fallback = "libx264"
    fallback_command = command.copy()
    codec_index = fallback_command.index("-c:v") + 1
    fallback_command[codec_index] = fallback
    bitrate_index = fallback_command.index("-b:v")
    fallback_command[bitrate_index : bitrate_index + 2] = ["-crf", "16"]
    run(fallback_command, cwd=cwd)
    return fallback


def mux_audio(visual_only: Path, audio: Path, output: Path, *, cwd: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(visual_only),
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output),
        ],
        cwd=cwd,
    )


def build_reel(root: Path, audio_path: Path, output_root: Path, run_id: str) -> dict:
    start = time.perf_counter()
    audio_duration = ffprobe_duration(audio_path, cwd=root)
    out_dir = output_root / run_id
    panels_dir = out_dir / "panels"
    bmp_dir = out_dir / "bmp_panels"
    segments_dir = out_dir / "segments"
    for directory in (panels_dir, bmp_dir, segments_dir):
        directory.mkdir(parents=True, exist_ok=True)

    edge = find_edge()
    profile = out_dir / "edge_profile"
    html = root / "ui" / "state_presentation_boardroom.html"
    panel_paths = []
    for slide in range(1, 13):
        panel_path = panels_dir / f"panel_{slide:02d}.png"
        export_panel(edge, html, panel_path, slide, profile)
        panel_paths.append(panel_path)
        run(["ffmpeg", "-y", "-i", str(panel_path), str(bmp_dir / f"panel_{slide:02d}.bmp")], cwd=root)

    clip_library = proof_clip_library(root)
    specs = system_specs(root)
    final_html = write_final_receipt_panel(out_dir, audio_path, audio_duration, clip_library, specs)
    final_png = panels_dir / "panel_final_receipt_hold.png"
    export_panel(edge, final_html, final_png, 1, profile)
    run(["ffmpeg", "-y", "-i", str(final_png), str(bmp_dir / "panel_final_receipt_hold.bmp")], cwd=root)

    base = [
        ("panel", 0, 12.0),
        ("panel", 1, 14.0),
        ("panel", 2, 12.0),
        ("panel", 3, 16.0),
        ("panel", 4, 16.0),
        ("clip", 0, 18.0),
        ("panel", 5, 14.0),
        ("clip", 1, 18.0),
        ("panel", 6, 14.0),
        ("panel", 7, 16.0),
        ("panel", 8, 18.0),
        ("clip", 2, 18.0),
        ("panel", 9, 14.0),
        ("panel", 10, 12.0),
    ]
    available_base = []
    for kind, index, seconds in base:
        if kind == "clip" and index >= len(clip_library):
            available_base.append(("panel", min(11, 8 + index), seconds))
        else:
            available_base.append((kind, index, seconds))

    total_segment_count = len(available_base) + 1
    total_transition_overlap = TRANSITION_SECONDS * max(0, total_segment_count - 1)
    final_segment_seconds = FINAL_RECEIPT_HOLD_SECONDS + TRANSITION_SECONDS
    pre_hold_target = max(10.0, audio_duration + total_transition_overlap - final_segment_seconds)
    base_total = sum(item[2] for item in available_base)
    scale = pre_hold_target / base_total
    segment_plan = [(kind, index, seconds * scale) for kind, index, seconds in available_base]
    segment_plan.append(("final", 0, final_segment_seconds))

    segments: list[Path] = []
    segment_durations: list[float] = []
    encoders_used = []
    for order, (kind, index, seconds) in enumerate(segment_plan, start=1):
        segment = segments_dir / f"segment_{order:02d}_{kind}.mp4"
        if kind == "panel":
            used = still_segment(panel_paths[index], segment, seconds, cwd=root, encoder="h264_qsv")
        elif kind == "clip":
            used = clip_segment(clip_library[index]["path"], segment, seconds, cwd=root, encoder="h264_qsv")
        else:
            used = still_segment(final_png, segment, seconds, cwd=root, encoder="h264_qsv")
        encoders_used.append(used)
        segments.append(segment)
        segment_durations.append(seconds)

    visual_only = out_dir / f"{run_id}_visual_only.mp4"
    output = out_dir / f"{run_id}.mp4"
    transition_encoder = crossfade_segments(
        segments,
        segment_durations,
        visual_only,
        cwd=root,
        encoder="h264_qsv",
    )
    encoders_used.append(f"{transition_encoder}_xfade")
    mux_audio(visual_only, audio_path, output, cwd=root)
    wall_seconds = time.perf_counter() - start
    output_duration = ffprobe_duration(output, cwd=root)

    manifest = {
        "schema": "truevision_gothic_industrial_proof_reel.v1",
        "run_id": run_id,
        "claim": "gothic-industrial proof reel built from systems panels, visible receipts, smooth transitions, and existing TrueVision media outputs",
        "boundary": {
            "synthetic_state_media_presentation": True,
            "not_evidence": True,
            "hide_backbone": True,
            "public_surface": "receipts, metrics, proof clips, system shape",
        },
        "input_audio": {
            "path": str(audio_path),
            "duration_seconds": audio_duration,
        },
        "render": {
            "output": str(output),
            "visual_only": str(visual_only),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_seconds": output_duration,
            "final_receipt_hold_seconds": FINAL_RECEIPT_HOLD_SECONDS,
            "transition_seconds": TRANSITION_SECONDS,
            "transition_style": "deterministic_crossfade",
            "panel_count": len(panel_paths),
            "segment_count": len(segments),
            "encoders_used": sorted(set(encoders_used)),
            "wall_seconds": wall_seconds,
            "speed_vs_realtime": output_duration / wall_seconds if wall_seconds else None,
        },
        "proof_clips": [
            {"id": clip["id"], "label": clip["label"], "path": str(clip["path"])}
            for clip in clip_library
        ],
        "artifacts": {
            "panels_png": str(panels_dir),
            "panels_bmp": str(bmp_dir),
            "segments": str(segments_dir),
            "final_receipt_panel": str(final_png),
        },
        "machine": specs,
        "third_party_surface": [
            "Microsoft Edge headless for local panel raster export",
            "FFmpeg and FFprobe for local media assembly and probing",
            "Local Windows fonts for typography",
        ],
    }
    manifest_path = out_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = out_dir / f"{run_id}_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# TrueVision Gothic Industrial Proof Reel",
                "",
                f"- Output: `{output}`",
                f"- Duration: {output_duration:.3f}s",
                f"- Audio: `{audio_path}`",
                f"- Final receipt hold: {FINAL_RECEIPT_HOLD_SECONDS:.1f}s",
                f"- Smooth transition length: {TRANSITION_SECONDS:.2f}s",
                f"- Panels exported: {len(panel_paths)} PNG and BMP copies",
                f"- Proof clips referenced: {len(clip_library)}",
                f"- Encoders used: {', '.join(sorted(set(encoders_used)))}",
                f"- Wall time: {wall_seconds:.3f}s",
                f"- Speed vs realtime: {manifest['render']['speed_vs_realtime']:.3f}x",
                "",
                "Visible receipts are shown in the final hold. Internal backbone details are not published.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TrueVision gothic-industrial proof reel panels and video.")
    parser.add_argument("--audio", required=True, help="Narration/music WAV file")
    parser.add_argument("--output-root", default="outputs/state_presentation", help="Output root")
    parser.add_argument("--run-id", default="state_media_boardroom_proof_reel", help="Run id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = build_reel(
        root=root,
        audio_path=Path(args.audio).resolve(),
        output_root=(root / args.output_root).resolve(),
        run_id=args.run_id,
    )
    print(json.dumps({"output": manifest["render"]["output"], "manifest": manifest["manifest_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
