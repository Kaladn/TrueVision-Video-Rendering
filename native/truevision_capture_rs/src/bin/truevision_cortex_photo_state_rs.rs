use std::env;
use std::fs::{create_dir_all, read_dir, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const STEMS: [(&str, &str, &str); 8] = [
    ("lead_vocals", "Lead Vocals", "main_title_charge_center_emblem"),
    ("backing_vocals", "Backing Vocals", "halo_echo_line_trace"),
    ("drums", "Drums", "impact_zoom_text_charge"),
    ("bass", "Bass", "shadow_pressure_depth_push"),
    ("guitar", "Guitar", "gold_edge_flame_circuit_glint"),
    ("percussion", "Percussion", "ember_tick_micro_scratch"),
    ("synth", "Synth", "smoke_density_negative_trace"),
    ("other", "Other", "background_grit_state_noise"),
];

#[derive(Clone)]
struct Args {
    output_root: PathBuf,
    run_id: String,
    audio: PathBuf,
    stems_dir: PathBuf,
    audio_source: String,
    image: PathBuf,
    plate_mode: String,
    width: usize,
    height: usize,
    fps: usize,
    duration: f64,
    video_encoder: String,
    bitrate: String,
    state_log_every: usize,
}

#[derive(Clone, Copy, Default)]
struct Meters {
    rms: f32,
    onset: f32,
    bass: f32,
    mid: f32,
    high: f32,
}

#[derive(Clone)]
struct StemTrack {
    id: &'static str,
    label: &'static str,
    lane: &'static str,
    source_path: PathBuf,
    activity_scale: f32,
    meters: Vec<Meters>,
}

#[derive(Clone, Copy, Default)]
struct Masks {
    gold: f32,
    text: f32,
    shadow: f32,
    smoke: f32,
    ember: f32,
    edge: f32,
    side_glyph_columns: f32,
    core_panel_lines: f32,
    sky_sun_glow: f32,
}

struct Plate {
    pixels: Vec<u8>,
    masks: Vec<Masks>,
    width: usize,
    height: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let started = Instant::now();
    let run_dir = args.output_root.join(&args.run_id);
    create_dir_all(&run_dir).map_err(|e| format!("create output dir failed: {e}"))?;

    let visual_path = run_dir.join(format!("{}_visual_only.mp4", args.run_id));
    let video_path = run_dir.join(format!("{}.mp4", args.run_id));
    let state_path = run_dir.join(format!("{}_frame_state.jsonl", args.run_id));
    let manifest_path = run_dir.join(format!("{}_manifest.json", args.run_id));
    let receipt_path = run_dir.join(format!("{}_receipt.json", args.run_id));
    let group_path = run_dir.join(format!("{}_artifact_groups.json", args.run_id));
    let frame_count = (args.duration * args.fps as f64).round().max(1.0) as usize;

    let tracks = if args.audio_source == "master_wav" {
        load_master_wav_tracks(&args.audio, args.duration, args.fps, frame_count)?
    } else {
        load_stem_tracks(&args.stems_dir, args.duration, args.fps, frame_count)?
    };
    let plate = decode_plate(&args.image, args.width, args.height, &args.plate_mode)?;
    File::create(&group_path)
        .map_err(|e| format!("artifact groups open failed: {e}"))?
        .write_all(artifact_groups_json(&args, &plate, &tracks).as_bytes())
        .map_err(|e| format!("artifact groups write failed: {e}"))?;

    let mut ffmpeg = Command::new("ffmpeg")
        .args(video_encode_args(&args, &visual_path))
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg start failed: {e}"))?;
    let mut stdin = ffmpeg
        .stdin
        .take()
        .ok_or_else(|| "ffmpeg stdin missing".to_string())?;
    let mut state_file =
        BufWriter::new(File::create(&state_path).map_err(|e| format!("state open failed: {e}"))?);
    let mut frame = vec![0_u8; args.width * args.height * 3];

    for frame_index in 0..frame_count {
        let time_seconds = frame_index as f64 / args.fps as f64;
        render_frame(&args, &plate, &tracks, frame_index, time_seconds as f32, &mut frame);
        stdin
            .write_all(&frame)
            .map_err(|e| format!("ffmpeg write failed: {e}"))?;
        if frame_index % args.state_log_every == 0 {
            writeln!(
                state_file,
                "{}",
                frame_state_json(&args, &tracks, frame_index, time_seconds)
            )
            .map_err(|e| format!("state write failed: {e}"))?;
        }
    }
    drop(stdin);
    let status = ffmpeg
        .wait()
        .map_err(|e| format!("ffmpeg wait failed: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg video encode failed with status {status}"));
    }
    state_file
        .flush()
        .map_err(|e| format!("state flush failed: {e}"))?;
    mux_audio(&visual_path, &args.audio, &video_path, args.duration)?;
    let _ = std::fs::remove_file(&visual_path);

    let wall_seconds = started.elapsed().as_secs_f64();
    File::create(&manifest_path)
        .map_err(|e| format!("manifest open failed: {e}"))?
        .write_all(
            manifest_json(
                &args,
                &video_path,
                &state_path,
                &group_path,
                frame_count,
                wall_seconds,
            )
            .as_bytes(),
        )
        .map_err(|e| format!("manifest write failed: {e}"))?;
    File::create(&receipt_path)
        .map_err(|e| format!("receipt open failed: {e}"))?
        .write_all(receipt_json(&args, &video_path, &manifest_path, &state_path, wall_seconds).as_bytes())
        .map_err(|e| format!("receipt write failed: {e}"))?;

    println!(
        "{{\"renderer\":\"rust\",\"preset\":\"cortex_photo_state_transform\",\"video_path\":\"{}\",\"manifest_path\":\"{}\",\"frame_count\":{},\"wall_seconds\":{:.3}}}",
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        frame_count,
        wall_seconds
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        output_root: PathBuf::from("outputs/cortex_photo_state_rs"),
        run_id: "cortex_photo_state_rs".to_string(),
        audio: PathBuf::new(),
        stems_dir: PathBuf::new(),
        audio_source: "stems".to_string(),
        image: PathBuf::new(),
        plate_mode: "portrait_fit".to_string(),
        width: 1080,
        height: 1920,
        fps: 30,
        duration: 30.0,
        video_encoder: "h264_qsv".to_string(),
        bitrate: "24M".to_string(),
        state_log_every: 30,
    };
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--output-root" => args.output_root = PathBuf::from(value),
            "--run-id" => args.run_id = slug(&value),
            "--audio" => args.audio = PathBuf::from(value),
            "--stems-dir" => args.stems_dir = PathBuf::from(value),
            "--audio-source" => args.audio_source = slug(&value),
            "--image" => args.image = PathBuf::from(value),
            "--plate-mode" => args.plate_mode = slug(&value),
            "--width" => args.width = value.parse().map_err(|_| "bad width".to_string())?,
            "--height" => args.height = value.parse().map_err(|_| "bad height".to_string())?,
            "--fps" => args.fps = value.parse().map_err(|_| "bad fps".to_string())?,
            "--duration" => args.duration = value.parse().map_err(|_| "bad duration".to_string())?,
            "--video-encoder" => args.video_encoder = slug(&value),
            "--bitrate" => args.bitrate = value,
            "--state-log-every" => {
                args.state_log_every = value
                    .parse()
                    .map_err(|_| "bad state log interval".to_string())?
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if args.audio.as_os_str().is_empty() {
        return Err("--audio is required".to_string());
    }
    if args.audio_source != "stems" && args.audio_source != "master_wav" {
        return Err("--audio-source must be stems or master_wav".to_string());
    }
    if args.audio_source == "stems" && args.stems_dir.as_os_str().is_empty() {
        return Err("--stems-dir is required".to_string());
    }
    if args.image.as_os_str().is_empty() {
        return Err("--image is required".to_string());
    }
    if args.width < 64 || args.height < 64 || args.fps < 1 || args.duration <= 0.0 || args.state_log_every < 1 {
        return Err("bad size, fps, duration, or log interval".to_string());
    }
    Ok(args)
}

fn load_stem_tracks(
    stems_dir: &Path,
    duration: f64,
    fps: usize,
    frame_count: usize,
) -> Result<Vec<StemTrack>, String> {
    let mut tracks = Vec::new();
    for (id, label, lane) in STEMS {
        let source_path = find_stem(stems_dir, label)?;
        let samples = decode_audio_with_ffmpeg(&source_path, duration)?;
        let activity_scale = stem_activity_scale(&samples);
        let meters = compute_meters(&samples, SAMPLE_RATE, fps, frame_count);
        tracks.push(StemTrack {
            id,
            label,
            lane,
            source_path,
            activity_scale,
            meters,
        });
    }
    Ok(tracks)
}

fn load_master_wav_tracks(
    audio_path: &Path,
    duration: f64,
    fps: usize,
    frame_count: usize,
) -> Result<Vec<StemTrack>, String> {
    let samples = decode_audio_with_ffmpeg(audio_path, duration)?;
    let base_scale = stem_activity_scale(&samples).max(0.35);
    let base = compute_meters(&samples, SAMPLE_RATE, fps, frame_count);
    let mut tracks = Vec::new();
    for (id, label, lane) in STEMS {
        let meters = derive_master_lane_meters(&base, id);
        tracks.push(StemTrack {
            id,
            label,
            lane,
            source_path: audio_path.to_path_buf(),
            activity_scale: base_scale,
            meters,
        });
    }
    Ok(tracks)
}

fn derive_master_lane_meters(base: &[Meters], id: &str) -> Vec<Meters> {
    base.iter()
        .map(|m| match id {
            "lead_vocals" => Meters {
                rms: (m.rms * 0.62 + m.mid * 0.38).clamp(0.0, 1.0),
                onset: m.onset * 0.55,
                bass: m.bass * 0.30,
                mid: (m.mid * 0.84 + m.rms * 0.16).clamp(0.0, 1.0),
                high: m.high * 0.34,
            },
            "backing_vocals" => Meters {
                rms: (m.rms * 0.40 + m.high * 0.20).clamp(0.0, 1.0),
                onset: m.onset * 0.35,
                bass: m.bass * 0.12,
                mid: m.mid * 0.48,
                high: m.high * 0.56,
            },
            "drums" => Meters {
                rms: (m.rms * 0.30 + m.onset * 0.60).clamp(0.0, 1.0),
                onset: m.onset,
                bass: (m.bass * 0.45 + m.onset * 0.30).clamp(0.0, 1.0),
                mid: m.mid * 0.30,
                high: (m.high * 0.55 + m.onset * 0.30).clamp(0.0, 1.0),
            },
            "bass" => Meters {
                rms: (m.rms * 0.35 + m.bass * 0.55).clamp(0.0, 1.0),
                onset: m.onset * 0.25,
                bass: m.bass,
                mid: m.mid * 0.14,
                high: m.high * 0.08,
            },
            "guitar" => Meters {
                rms: (m.rms * 0.32 + m.mid * 0.36 + m.high * 0.20).clamp(0.0, 1.0),
                onset: m.onset * 0.48,
                bass: m.bass * 0.16,
                mid: m.mid * 0.78,
                high: m.high * 0.62,
            },
            "percussion" => Meters {
                rms: (m.onset * 0.55 + m.high * 0.35).clamp(0.0, 1.0),
                onset: m.onset,
                bass: m.bass * 0.08,
                mid: m.mid * 0.28,
                high: m.high * 0.82,
            },
            "synth" => Meters {
                rms: (m.rms * 0.55 + m.high * 0.18).clamp(0.0, 1.0),
                onset: m.onset * 0.20,
                bass: m.bass * 0.24,
                mid: m.mid * 0.44,
                high: m.high * 0.70,
            },
            _ => Meters {
                rms: m.rms * 0.42,
                onset: m.onset * 0.22,
                bass: m.bass * 0.36,
                mid: m.mid * 0.30,
                high: m.high * 0.22,
            },
        })
        .collect()
}

fn find_stem(stems_dir: &Path, label: &str) -> Result<PathBuf, String> {
    let needle = label.to_ascii_lowercase();
    let compact_needle = needle.replace(' ', "");
    let mut candidates = Vec::new();
    for entry in read_dir(stems_dir).map_err(|e| format!("read stems dir failed: {e}"))? {
        let path = entry.map_err(|e| format!("read stem entry failed: {e}"))?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name_os) = path.file_name() else {
            continue;
        };
        let name = name_os.to_string_lossy().to_ascii_lowercase();
        let compact_name = name.replace([' ', '_', '-'], "");
        let supported = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
            .iter()
            .any(|ext| name.ends_with(ext));
        if supported && (name.contains(&needle) || compact_name.contains(&compact_needle)) {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| format!("missing stem file for {label}"))
}

fn decode_audio_with_ffmpeg(path: &Path, duration: f64) -> Result<Vec<f32>, String> {
    let output = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-t")
        .arg(format!("{duration:.6}"))
        .arg("-i")
        .arg(path)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(SAMPLE_RATE.to_string())
        .arg("-f")
        .arg("f32le")
        .arg("-")
        .output()
        .map_err(|e| format!("ffmpeg decode failed for {}: {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg decode failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let mut samples = Vec::with_capacity(output.stdout.len() / 4);
    for chunk in output.stdout.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0));
    }
    let target = (duration * SAMPLE_RATE as f64).round().max(1.0) as usize;
    samples.resize(target, 0.0);
    samples.truncate(target);
    Ok(samples)
}

fn stem_activity_scale(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0_f32;
    let mut peak = 0.0_f32;
    for sample in samples {
        let abs = sample.abs();
        peak = peak.max(abs);
        sum_sq += sample * sample;
    }
    let rms = (sum_sq / samples.len() as f32).sqrt();
    ((rms * 13.0).max(peak * 1.6)).clamp(0.05, 1.0)
}

fn compute_meters(samples: &[f32], sample_rate: usize, fps: usize, frame_count: usize) -> Vec<Meters> {
    let hop = (sample_rate as f64 / fps as f64).round().max(1.0) as usize;
    let mut raw = vec![Meters::default(); frame_count];
    for (frame_index, meter) in raw.iter_mut().enumerate() {
        let start = frame_index * hop;
        let end = (start + hop).min(samples.len());
        if start >= end {
            continue;
        }
        let mut sum_sq = 0.0_f32;
        let mut bass_sq = 0.0_f32;
        let mut mid_sq = 0.0_f32;
        let mut high_sq = 0.0_f32;
        let mut low = 0.0_f32;
        let mut prev = samples[start];
        let mut prev_diff = 0.0_f32;
        for sample in &samples[start..end] {
            let s = *sample;
            low = low * 0.986 + s * 0.014;
            let diff = s - prev;
            let diff2 = diff - prev_diff;
            sum_sq += s * s;
            bass_sq += low * low;
            mid_sq += (s - low) * (s - low);
            high_sq += diff2 * diff2;
            prev = s;
            prev_diff = diff;
        }
        let n = (end - start) as f32;
        meter.rms = (sum_sq / n).sqrt();
        meter.bass = (bass_sq / n).sqrt();
        meter.mid = (mid_sq / n).sqrt();
        meter.high = (high_sq / n).sqrt();
    }
    for index in 1..frame_count {
        raw[index].onset = ((raw[index].rms - raw[index - 1].rms) * 3.0).max(0.0);
    }
    normalize_meter_lanes(&mut raw);
    raw
}

fn normalize_meter_lanes(meters: &mut [Meters]) {
    let mut maxes = Meters::default();
    for meter in meters.iter() {
        maxes.rms = maxes.rms.max(meter.rms);
        maxes.onset = maxes.onset.max(meter.onset);
        maxes.bass = maxes.bass.max(meter.bass);
        maxes.mid = maxes.mid.max(meter.mid);
        maxes.high = maxes.high.max(meter.high);
    }
    for meter in meters.iter_mut() {
        meter.rms = norm(meter.rms, maxes.rms);
        meter.onset = norm(meter.onset, maxes.onset);
        meter.bass = norm(meter.bass, maxes.bass);
        meter.mid = norm(meter.mid, maxes.mid);
        meter.high = norm(meter.high, maxes.high);
    }
}

fn norm(value: f32, max_value: f32) -> f32 {
    if max_value <= 0.000_001 {
        0.0
    } else {
        (value / max_value).clamp(0.0, 1.0)
    }
}

fn scaled_meters_for(tracks: &[StemTrack], id: &str, frame_index: usize) -> Meters {
    let Some(track) = tracks.iter().find(|track| track.id == id) else {
        return Meters::default();
    };
    let mut meters = track.meters.get(frame_index).copied().unwrap_or_default();
    meters.rms *= track.activity_scale;
    meters.onset *= track.activity_scale;
    meters.bass *= track.activity_scale;
    meters.mid *= track.activity_scale;
    meters.high *= track.activity_scale;
    meters
}

fn decode_plate(path: &Path, width: usize, height: usize, plate_mode: &str) -> Result<Plate, String> {
    let filter = if plate_mode == "portrait_crop" {
        format!(
            "scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        )
    } else {
        format!(
            "scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    };
    let output = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-vf")
        .arg(filter)
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-")
        .output()
        .map_err(|e| format!("ffmpeg image decode failed: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg image decode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let expected = width * height * 3;
    if output.stdout.len() != expected {
        return Err(format!(
            "image decode size mismatch: got {}, expected {}",
            output.stdout.len(),
            expected
        ));
    }
    let masks = compute_masks(&output.stdout, width, height);
    Ok(Plate {
        pixels: output.stdout,
        masks,
        width,
        height,
    })
}

fn compute_masks(pixels: &[u8], width: usize, height: usize) -> Vec<Masks> {
    let mut masks = vec![Masks::default(); width * height];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            let r = pixels[i] as f32 / 255.0;
            let g = pixels[i + 1] as f32 / 255.0;
            let b = pixels[i + 2] as f32 / 255.0;
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = max_c - min_c;
            let gold = smoothstep(0.08, 0.54, r - b) * smoothstep(0.02, 0.42, g - b) * smoothstep(0.18, 0.86, lum);
            let text = smoothstep(0.50, 0.88, lum) * smoothstep(0.0, 0.36, 1.0 - sat);
            let shadow = smoothstep(0.38, 0.02, lum);
            let smoke = smoothstep(0.16, 0.58, lum) * smoothstep(0.0, 0.32, 1.0 - sat);
            let ember = smoothstep(0.18, 0.86, r) * smoothstep(0.00, 0.36, g) * smoothstep(0.0, 0.18, b);
            let edge = local_edge(pixels, width, height, x, y);
            let nx = if width > 1 { x as f32 / (width - 1) as f32 } else { 0.0 };
            let ny = if height > 1 { y as f32 / (height - 1) as f32 } else { 0.0 };
            let gold_or_line = (gold * 0.78 + text * 0.34 + edge * 0.54).clamp(0.0, 1.0);
            let side_window = smoothstep(0.16, 0.30, ny) * (1.0 - smoothstep(0.72, 0.88, ny));
            let left_column = gaussian(nx, 0.145, 0.026);
            let right_column = gaussian(nx, 0.855, 0.026);
            let side_glyph_columns = (left_column + right_column).clamp(0.0, 1.0) * side_window * gold_or_line;
            let core_window = smoothstep(0.20, 0.34, ny) * (1.0 - smoothstep(0.58, 0.76, ny));
            let core_left_line = gaussian(nx, 0.430, 0.014);
            let core_right_line = gaussian(nx, 0.570, 0.014);
            let core_panel_lines = (core_left_line + core_right_line).clamp(0.0, 1.0) * core_window * gold_or_line;
            let horizon_window = smoothstep(0.45, 0.62, ny) * (1.0 - smoothstep(0.88, 0.99, ny));
            let horizon_center = gaussian(nx, 0.50, 0.34);
            let warm_sky = smoothstep(0.10, 0.62, r - b) * smoothstep(0.14, 0.76, lum);
            let sky_sun_glow = horizon_window * horizon_center * warm_sky * (1.0 - text * 0.70);
            masks[y * width + x] = Masks {
                gold,
                text,
                shadow,
                smoke,
                ember,
                edge,
                side_glyph_columns,
                core_panel_lines,
                sky_sun_glow,
            };
        }
    }
    masks
}

fn gaussian(value: f32, center: f32, sigma: f32) -> f32 {
    let d = (value - center) / sigma.max(0.0001);
    (-0.5 * d * d).exp().clamp(0.0, 1.0)
}

fn local_edge(pixels: &[u8], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let lum_at = |xx: usize, yy: usize| -> f32 {
        let i = (yy * width + xx) * 3;
        (pixels[i] as f32 * 0.2126 + pixels[i + 1] as f32 * 0.7152 + pixels[i + 2] as f32 * 0.0722) / 255.0
    };
    let x0 = x.saturating_sub(1);
    let x1 = (x + 1).min(width.saturating_sub(1));
    let y0 = y.saturating_sub(1);
    let y1 = (y + 1).min(height.saturating_sub(1));
    let gx = (lum_at(x1, y) - lum_at(x0, y)).abs();
    let gy = (lum_at(x, y1) - lum_at(x, y0)).abs();
    ((gx + gy) * 2.4).clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if (edge1 - edge0).abs() < 0.0001 {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn render_frame(args: &Args, plate: &Plate, tracks: &[StemTrack], frame_index: usize, t: f32, frame: &mut [u8]) {
    let lead = scaled_meters_for(tracks, "lead_vocals", frame_index);
    let backing = scaled_meters_for(tracks, "backing_vocals", frame_index);
    let drums = scaled_meters_for(tracks, "drums", frame_index);
    let bass = scaled_meters_for(tracks, "bass", frame_index);
    let guitar = scaled_meters_for(tracks, "guitar", frame_index);
    let percussion = scaled_meters_for(tracks, "percussion", frame_index);
    let synth = scaled_meters_for(tracks, "synth", frame_index);
    let other = scaled_meters_for(tracks, "other", frame_index);

    let cam = camera_pressure(drums, bass, guitar, lead, synth, t);
    let cx = plate.width as f32 * 0.5;
    let cy = plate.height as f32 * 0.5;
    let cos_r = cam.rotation.sin_cos().1;
    let sin_r = cam.rotation.sin_cos().0;

    for y in 0..args.height {
        for x in 0..args.width {
            let fx = x as f32 - cx;
            let fy = y as f32 - cy;
            let px = (fx / cam.zoom) - cam.pan_x * args.width as f32;
            let py = (fy / cam.zoom) - cam.pan_y * args.height as f32;
            let sx = (px * cos_r + py * sin_r + cx).round() as i32;
            let sy = (-px * sin_r + py * cos_r + cy).round() as i32;
            let out_i = (y * args.width + x) * 3;
            if sx < 0 || sy < 0 || sx >= plate.width as i32 || sy >= plate.height as i32 {
                frame[out_i] = 0;
                frame[out_i + 1] = 0;
                frame[out_i + 2] = 0;
                continue;
            }
            let si = (sy as usize * plate.width + sx as usize) * 3;
            let mi = sy as usize * plate.width + sx as usize;
            let m = plate.masks[mi];
            let mut r = plate.pixels[si] as f32;
            let mut g = plate.pixels[si + 1] as f32;
            let mut b = plate.pixels[si + 2] as f32;

            let shimmer = 0.5 + 0.5 * (sx as f32 * 0.031 + sy as f32 * 0.017 + t * (3.0 + guitar.mid * 5.0)).sin();
            let gold_flame = m.gold * (0.10 + guitar.mid * 0.36 + percussion.high * 0.13) * shimmer;
            r += gold_flame * 106.0;
            g += gold_flame * 58.0;
            b += gold_flame * 8.0;

            let text_charge = m.text * (0.05 + lead.mid * 0.20 + drums.onset * 0.20);
            r += text_charge * 54.0;
            g += text_charge * 52.0;
            b += text_charge * 48.0;

            let shadow_pressure = m.shadow * (0.05 + bass.bass * 0.18 + other.rms * 0.08);
            r *= 1.0 - shadow_pressure * 0.38;
            g *= 1.0 - shadow_pressure * 0.34;
            b *= 1.0 - shadow_pressure * 0.26;

            let smoke_wave = m.smoke * (0.03 + synth.rms * 0.16) * (0.6 + 0.4 * (t * 0.8 + sx as f32 * 0.005).sin());
            r += smoke_wave * 28.0;
            g += smoke_wave * 26.0;
            b += smoke_wave * 34.0;

            let sun_breath = m.sky_sun_glow
                * (0.035 + bass.bass * 0.070 + lead.rms * 0.055 + synth.rms * 0.035)
                * (0.74 + 0.26 * (t * 0.72).sin());
            let sun_cap = 232.0;
            r = (r + sun_breath * 86.0).min(sun_cap);
            g = (g + sun_breath * 48.0).min(sun_cap * 0.86);
            b = (b + sun_breath * 14.0).min(sun_cap * 0.48);

            let edge_trace = m.edge * (0.035 + synth.high * 0.22 + backing.mid * 0.11);
            let negative = 255.0 - (r + g + b) / 3.0;
            r += edge_trace * negative * 0.75;
            g += edge_trace * negative * 0.55;
            b += edge_trace * negative * 0.28;

            let ember_tick = m.ember * (0.06 + percussion.onset * 0.45 + drums.onset * 0.20);
            r += ember_tick * 112.0;
            g += ember_tick * 44.0;
            b += ember_tick * 4.0;

            let side_column_pulse = m.side_glyph_columns
                * (0.08 + backing.mid * 0.18 + synth.high * 0.16 + guitar.high * 0.08)
                * (0.65 + 0.35 * (t * 2.2 + sy as f32 * 0.031).sin());
            r += side_column_pulse * 118.0;
            g += side_column_pulse * 78.0;
            b += side_column_pulse * 16.0;

            let core_line_charge = m.core_panel_lines
                * (0.10 + lead.mid * 0.22 + drums.onset * 0.24 + bass.bass * 0.08)
                * (0.70 + 0.30 * (t * 3.1 + sx as f32 * 0.018).cos());
            r += core_line_charge * 96.0;
            g += core_line_charge * 92.0;
            b += core_line_charge * 64.0;

            frame[out_i] = r.clamp(0.0, 255.0) as u8;
            frame[out_i + 1] = g.clamp(0.0, 255.0) as u8;
            frame[out_i + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
}

#[derive(Clone, Copy)]
struct CameraPressure {
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    rotation: f32,
}

fn camera_pressure(drums: Meters, bass: Meters, guitar: Meters, lead: Meters, synth: Meters, t: f32) -> CameraPressure {
    let zoom = 1.006 + bass.bass * 0.020 + drums.onset * 0.018 + lead.rms * 0.008;
    let pan_x = (t * 0.19 + guitar.mid * 1.2).sin() * (0.0025 + guitar.mid * 0.0045);
    let pan_y = (t * 0.15 + synth.rms).cos() * (0.0020 + bass.bass * 0.0035);
    let rotation = (t * 0.34).sin() * (0.0022 + guitar.rms * 0.0045 + drums.onset * 0.0040);
    CameraPressure {
        zoom: zoom.min(1.055),
        pan_x,
        pan_y,
        rotation,
    }
}

fn artifact_groups_json(args: &Args, plate: &Plate, tracks: &[StemTrack]) -> String {
    let mut sums = Masks::default();
    for mask in &plate.masks {
        sums.gold += mask.gold;
        sums.text += mask.text;
        sums.shadow += mask.shadow;
        sums.smoke += mask.smoke;
        sums.ember += mask.ember;
        sums.edge += mask.edge;
        sums.side_glyph_columns += mask.side_glyph_columns;
        sums.core_panel_lines += mask.core_panel_lines;
        sums.sky_sun_glow += mask.sky_sun_glow;
    }
    let total = plate.masks.len().max(1) as f32;
    let mut stem_sources = String::new();
    for (index, track) in tracks.iter().enumerate() {
        if index > 0 {
            stem_sources.push(',');
        }
        stem_sources.push_str(&format!(
            "{{\"stem_id\":\"{}\",\"label\":\"{}\",\"source_path\":\"{}\",\"lane\":\"{}\",\"activity_scale\":{:.6}}}",
            track.id,
            track.label,
            json_escape(&track.source_path.display().to_string()),
            track.lane,
            track.activity_scale
        ));
    }
    format!(
        "{{\n  \"schema_version\":\"cortex_photo_artifact_groups_v1\",\n  \"source_image\":\"{}\",\n  \"artifact_groups\":{{\"gold_items\":{:.6},\"text_glyphs\":{:.6},\"shadow_regions\":{:.6},\"smoke_atmosphere\":{:.6},\"ember_points\":{:.6},\"line_art_edges\":{:.6},\"side_glyph_columns\":{:.6},\"core_panel_lines\":{:.6},\"sky_sun_glow\":{:.6}}},\n  \"stem_sources\":[{}],\n  \"glyph_schema\":\"artwork_color_schema_gold_white_on_black\",\n  \"plate_mode\":\"{}\",\n  \"audio_source\":\"{}\"\n}}\n",
        json_escape(&args.image.display().to_string()),
        sums.gold / total,
        sums.text / total,
        sums.shadow / total,
        sums.smoke / total,
        sums.ember / total,
        sums.edge / total,
        sums.side_glyph_columns / total,
        sums.core_panel_lines / total,
        sums.sky_sun_glow / total,
        stem_sources,
        json_escape(&args.plate_mode),
        json_escape(&args.audio_source)
    )
}

fn frame_state_json(args: &Args, tracks: &[StemTrack], frame_index: usize, time_seconds: f64) -> String {
    let mut controls = String::new();
    for (index, track) in tracks.iter().enumerate() {
        if index > 0 {
            controls.push(',');
        }
        let m = scaled_meters_for(tracks, track.id, frame_index);
        controls.push_str(&format!(
            "\"{}\":{{\"label\":\"{}\",\"lane\":\"{}\",\"activity_scale\":{:.6},\"meters\":{{\"rms\":{:.6},\"onset\":{:.6},\"bass\":{:.6},\"mid\":{:.6},\"high\":{:.6}}}}}",
            track.id, track.label, track.lane, track.activity_scale, m.rms, m.onset, m.bass, m.mid, m.high
        ));
    }
    format!(
        "{{\"schema_version\":\"cortex_photo_state_frame_rs_v1\",\"renderer\":\"rust\",\"preset\":\"cortex_photo_state_transform\",\"frame_index\":{},\"time_seconds\":{:.6},\"source_image\":\"{}\",\"audio_source\":\"{}\",\"stem_controls\":{{{}}},\"artifact_groups\":[\"gold_items\",\"text_glyphs\",\"shadow_regions\",\"smoke_atmosphere\",\"ember_points\",\"line_art_edges\",\"side_glyph_columns\",\"core_panel_lines\",\"sky_sun_glow\"],\"audio_meter_bound_artifact_groups\":[\"side_glyph_columns\",\"core_panel_lines\",\"sky_sun_glow\"],\"elemental_transforms\":[\"gold_edge_flame\",\"white_text_charge\",\"shadow_pressure\",\"smoke_density_wave\",\"ember_tick\",\"line_art_negative_trace\",\"side_glyph_column_pulse\",\"core_panel_line_charge\",\"sky_sun_illumination\",\"camera_pressure\"],\"technical_provenance\":{{\"hardware\":\"Intel Arc / QSV encode\",\"software\":\"Rust TrueVision + FFmpeg\",\"public_method\":\"source-photo-pixel-state transform\",\"glyph_schema\":\"artwork_color_schema_gold_white_on_black\",\"artwork_color_schema\":\"gold_white_on_black\"}},\"boundary\":{{\"photo_state_transform\":true,\"source_pixel_transform_only\":true,\"no_added_visual_artifacts\":true,\"glyph_schema_from_artwork\":true,\"technical_identity_banner\":false,\"moving_objects\":false,\"generated_media_is_evidence\":false}}}}",
        frame_index,
        time_seconds,
        json_escape(&args.image.display().to_string()),
        json_escape(&args.audio_source),
        controls
    )
}

fn manifest_json(
    args: &Args,
    video_path: &PathBuf,
    state_path: &PathBuf,
    group_path: &PathBuf,
    frame_count: usize,
    wall_seconds: f64,
) -> String {
    format!(
        "{{\n  \"schema_version\":\"cortex_photo_state_manifest_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"cortex_photo_state_transform\",\n  \"run_id\":\"{}\",\n  \"source\":{{\"master_audio\":\"{}\",\"stems_dir\":\"{}\",\"audio_source\":\"{}\",\"image\":\"{}\",\"plate_mode\":\"{}\"}},\n  \"output\":{{\"mp4\":\"{}\",\"frame_state_jsonl\":\"{}\",\"artifact_groups_json\":\"{}\",\"width\":{},\"height\":{},\"fps\":{},\"duration_seconds\":{},\"frame_count\":{},\"encoder\":\"{}\",\"wall_seconds\":{:.6}}},\n  \"visual_contract\":{{\"method\":\"source_photo_pixel_state_transform\",\"hardware\":\"Intel Arc / h264_qsv encode\",\"software\":\"Rust TrueVision + FFmpeg\",\"public_method\":\"source photo artifact groups plus audio meters transform existing pixels only\",\"glyph_schema\":\"artwork_color_schema_gold_white_on_black\"}},\n  \"boundary\":{{\"photo_state_transform\":true,\"source_pixel_transform_only\":true,\"no_added_visual_artifacts\":true,\"glyph_schema_from_artwork\":true,\"technical_identity_banner\":false,\"moving_objects\":false,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_dir.display().to_string()),
        json_escape(&args.audio_source),
        json_escape(&args.image.display().to_string()),
        json_escape(&args.plate_mode),
        json_escape(&video_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        json_escape(&group_path.display().to_string()),
        args.width,
        args.height,
        args.fps,
        args.duration,
        frame_count,
        json_escape(&args.video_encoder),
        wall_seconds
    )
}

fn receipt_json(args: &Args, video_path: &PathBuf, manifest_path: &PathBuf, state_path: &PathBuf, wall_seconds: f64) -> String {
    format!(
        "{{\n  \"schema_version\":\"cortex_photo_state_receipt_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"cortex_photo_state_transform\",\n  \"run_id\":\"{}\",\n  \"status\":\"complete\",\n  \"audio_source\":\"{}\",\n  \"output_mp4\":\"{}\",\n  \"manifest_json\":\"{}\",\n  \"frame_state_jsonl\":\"{}\",\n  \"elapsed_seconds\":{:.6},\n  \"boundary\":{{\"photo_state_transform\":true,\"source_pixel_transform_only\":true,\"no_added_visual_artifacts\":true,\"glyph_schema_from_artwork\":true,\"technical_identity_banner\":false,\"moving_objects\":false,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&args.audio_source),
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        wall_seconds
    )
}

fn video_encode_args(args: &Args, video_path: &PathBuf) -> Vec<String> {
    let mut out = vec![
        "-y".to_string(),
        "-v".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-s".to_string(),
        format!("{}x{}", args.width, args.height),
        "-r".to_string(),
        args.fps.to_string(),
        "-i".to_string(),
        "-".to_string(),
        "-an".to_string(),
    ];
    if args.video_encoder == "libx264" {
        out.extend([
            "-c:v".to_string(),
            "libx264".to_string(),
            "-preset".to_string(),
            "veryfast".to_string(),
            "-crf".to_string(),
            "16".to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
        ]);
    } else {
        out.extend([
            "-vf".to_string(),
            "format=nv12".to_string(),
            "-c:v".to_string(),
            args.video_encoder.clone(),
            "-b:v".to_string(),
            args.bitrate.clone(),
            "-maxrate".to_string(),
            args.bitrate.clone(),
            "-pix_fmt".to_string(),
            "nv12".to_string(),
        ]);
    }
    out.push(video_path.display().to_string());
    out
}

fn mux_audio(video_path: &PathBuf, audio_path: &PathBuf, output_path: &PathBuf, duration: f64) -> Result<(), String> {
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(video_path)
        .arg("-t")
        .arg(format!("{duration:.6}"))
        .arg("-i")
        .arg(audio_path)
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("1:a:0")
        .arg("-c:v")
        .arg("copy")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("192k")
        .arg("-shortest")
        .arg(output_path)
        .status()
        .map_err(|e| format!("audio mux failed: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("audio mux failed with status {status}"))
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn slug(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch);
        } else if ch.is_whitespace() || matches!(ch, '.' | '/' | '\\') {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    out.trim_matches('_').to_string()
}
