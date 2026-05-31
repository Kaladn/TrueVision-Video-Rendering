use std::env;
use std::fs::{create_dir_all, read_dir, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const STEMS: [(&str, &str, &str); 12] = [
    ("lead_vocals", "Lead Vocals", "lead_vocal_ribbon"),
    ("backing_vocals", "Backing Vocals", "orbit_echoes"),
    ("drums", "Drums", "controlled_grid_pulses"),
    ("bass", "Bass", "floor_depth_waves"),
    ("guitar", "Guitar", "warm_horizontal_trails"),
    ("keyboard", "Keyboard", "keyboard_harmonic_lattice"),
    ("percussion", "Percussion", "tiny_tick_sparks"),
    ("strings", "Strings", "strings_fate_arcs"),
    ("synth", "Synth", "ambient_halo_depth_haze"),
    ("other", "Other", "low_opacity_texture"),
    ("brass", "Brass", "rare_royal_accent_flares"),
    ("woodwinds", "Woodwinds", "rare_royal_accent_flares"),
];

#[derive(Clone)]
struct Args {
    output_root: PathBuf,
    run_id: String,
    audio: PathBuf,
    stems_dir: PathBuf,
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

#[derive(Clone, Copy)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl Color {
    fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
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
    let meter_path = run_dir.join(format!("{}_stem_meter_summary.json", args.run_id));
    let manifest_path = run_dir.join(format!("{}_manifest.json", args.run_id));
    let receipt_path = run_dir.join(format!("{}_receipt.json", args.run_id));
    let frame_count = (args.duration * args.fps as f64).round().max(1.0) as usize;

    let tracks = load_stem_tracks(&args.stems_dir, args.duration, args.fps, frame_count)?;
    File::create(&meter_path)
        .map_err(|e| format!("meter summary open failed: {e}"))?
        .write_all(build_meter_summary(&args, &tracks, frame_count).as_bytes())
        .map_err(|e| format!("meter summary write failed: {e}"))?;

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
        render_frame(&args, &tracks, frame_index, time_seconds as f32, &mut frame);
        stdin
            .write_all(&frame)
            .map_err(|e| format!("ffmpeg write failed: {e}"))?;
        if frame_index % args.state_log_every == 0 {
            writeln!(state_file, "{}", frame_state_json(&tracks, frame_index, time_seconds))
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
            manifest_json(&args, &video_path, &state_path, &meter_path, frame_count, wall_seconds)
                .as_bytes(),
        )
        .map_err(|e| format!("manifest write failed: {e}"))?;
    File::create(&receipt_path)
        .map_err(|e| format!("receipt open failed: {e}"))?
        .write_all(receipt_json(&args, &video_path, &manifest_path, &state_path, wall_seconds).as_bytes())
        .map_err(|e| format!("receipt write failed: {e}"))?;

    println!(
        "{{\"renderer\":\"rust\",\"preset\":\"distant_love_state_soft\",\"video_path\":\"{}\",\"manifest_path\":\"{}\",\"frame_count\":{},\"wall_seconds\":{:.3}}}",
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        frame_count,
        wall_seconds
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        output_root: PathBuf::from("outputs/distant_love_state_soft_rs"),
        run_id: "distant_love_state_soft_rs".to_string(),
        audio: PathBuf::new(),
        stems_dir: PathBuf::new(),
        width: 1280,
        height: 720,
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
    if args.stems_dir.as_os_str().is_empty() {
        return Err("--stems-dir is required".to_string());
    }
    if args.width < 64 || args.height < 64 {
        return Err("size too small".to_string());
    }
    if args.fps < 1 || args.duration <= 0.0 || args.state_log_every < 1 {
        return Err("fps, duration, and state-log-every must be positive".to_string());
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
    ((rms * 14.0).max(peak * 1.7)).clamp(0.06, 1.0)
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
            low = low * 0.987 + s * 0.013;
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

fn render_frame(args: &Args, tracks: &[StemTrack], frame_index: usize, t: f32, frame: &mut [u8]) {
    let lead = scaled_meters_for(tracks, "lead_vocals", frame_index);
    let backing = scaled_meters_for(tracks, "backing_vocals", frame_index);
    let drums = scaled_meters_for(tracks, "drums", frame_index);
    let bass = scaled_meters_for(tracks, "bass", frame_index);
    let guitar = scaled_meters_for(tracks, "guitar", frame_index);
    let keyboard = scaled_meters_for(tracks, "keyboard", frame_index);
    let percussion = scaled_meters_for(tracks, "percussion", frame_index);
    let strings = scaled_meters_for(tracks, "strings", frame_index);
    let synth = scaled_meters_for(tracks, "synth", frame_index);
    let other = scaled_meters_for(tracks, "other", frame_index);
    let brass = scaled_meters_for(tracks, "brass", frame_index);
    let woodwinds = scaled_meters_for(tracks, "woodwinds", frame_index);

    fill_soft_background(frame, args.width, args.height, synth, bass, t);
    draw_depth_grid(frame, args.width, args.height, bass, drums, t);
    draw_keyboard_lattice(frame, args.width, args.height, keyboard, t);
    draw_castle_line_art(frame, args.width, args.height, brass, woodwinds, t);
    draw_battlefield_floor_waves(frame, args.width, args.height, bass, guitar, t);
    draw_strings_fate_arcs(frame, args.width, args.height, strings, t);
    draw_dual_distance_traces(frame, args.width, args.height, lead, backing, strings, t);
    draw_synth_halo_depth_haze(frame, args.width, args.height, synth, other, t);
    draw_lead_vocal_ribbon(frame, args.width, args.height, lead, t);
    draw_backing_vocal_orbits(frame, args.width, args.height, backing, t);
    draw_guitar_warm_trails(frame, args.width, args.height, guitar, t);
    draw_drums_and_percussion(frame, args.width, args.height, drums, percussion, frame_index, t);
    draw_soft_stem_meter_overlay(frame, args.width, args.height, tracks, frame_index);
}

fn fill_soft_background(frame: &mut [u8], width: usize, height: usize, synth: Meters, bass: Meters, t: f32) {
    for y in 0..height {
        let yy = y as f32 / height.max(1) as f32;
        let pulse = 0.5 + 0.5 * (t * 0.18).sin();
        let r = (0.014 + yy * (0.020 + bass.bass * 0.022) + synth.high * 0.010 * pulse) * 255.0;
        let g = (0.014 + yy * 0.012 + synth.rms * 0.010) * 255.0;
        let b = (0.026 + (1.0 - yy) * (0.032 + synth.mid * 0.024)) * 255.0;
        for x in 0..width {
            let i = (y * width + x) * 3;
            frame[i] = r as u8;
            frame[i + 1] = g as u8;
            frame[i + 2] = b as u8;
        }
    }
}

fn draw_depth_grid(frame: &mut [u8], width: usize, height: usize, bass: Meters, drums: Meters, t: f32) {
    let horizon = (height as f32 * (0.57 + 0.018 * (t * 0.33).sin())) as i32;
    let vanish = (width as i32 / 2, horizon);
    let color = Color::new(90.0, 70.0 + bass.bass * 42.0, 62.0);
    for line in 0..17 {
        let x = (-0.18 * width as f32 + (line as f32 / 16.0) * width as f32 * 1.36) as i32;
        draw_line(frame, width, height, vanish.0, vanish.1, x, height as i32, color, 0.07 + drums.rms * 0.06, 1);
    }
    for row in 0..12 {
        let n = row as f32 / 11.0;
        let y = horizon as f32 + n.powf(1.85) * (height as f32 - horizon as f32);
        draw_line(frame, width, height, 0, y as i32, width as i32, y as i32, color, 0.08 + bass.bass * 0.05, 1);
    }
}

fn draw_keyboard_lattice(frame: &mut [u8], width: usize, height: usize, keyboard: Meters, t: f32) {
    let color = Color::new(92.0, 116.0, 140.0);
    let y_mid = height as f32 * 0.43;
    for index in 0..9 {
        let n = index as f32 / 8.0;
        let y = y_mid + (n - 0.5) * height as f32 * 0.32 + (t * 0.7 + index as f32).sin() * keyboard.mid * 9.0;
        draw_line(frame, width, height, (width as f32 * 0.12) as i32, y as i32, (width as f32 * 0.88) as i32, y as i32, color, 0.045 + keyboard.rms * 0.080, 1);
    }
    for index in 0..10 {
        let x = (width as f32 * (0.16 + index as f32 * 0.075)) as i32;
        draw_line(frame, width, height, x, (height as f32 * 0.26) as i32, x, (height as f32 * 0.62) as i32, color, 0.030 + keyboard.high * 0.050, 1);
    }
}

fn draw_castle_line_art(frame: &mut [u8], width: usize, height: usize, brass: Meters, woodwinds: Meters, t: f32) {
    let cx = width as i32 / 2;
    let base_y = (height as f32 * 0.50) as i32;
    let w = (width as f32 * 0.25) as i32;
    let h = (height as f32 * 0.20) as i32;
    let lift = (brass.rms.max(woodwinds.rms) * 20.0) as i32;
    let white = Color::new(198.0 + brass.rms * 40.0, 210.0 + woodwinds.rms * 35.0, 224.0);
    let alpha = 0.16 + brass.rms.max(woodwinds.rms) * 0.16;
    draw_line(frame, width, height, cx - w, base_y, cx + w, base_y, white, alpha, 1);
    draw_line(frame, width, height, cx - w, base_y, cx - w, base_y - h, white, alpha, 1);
    draw_line(frame, width, height, cx + w, base_y, cx + w, base_y - h, white, alpha, 1);
    draw_line(frame, width, height, cx - w, base_y - h, cx + w, base_y - h, white, alpha, 1);
    for tower in [-1, 0, 1] {
        let tx = cx + tower * w / 2;
        let th = h + if tower == 0 { h / 2 } else { h / 4 } + lift;
        draw_line(frame, width, height, tx - w / 10, base_y, tx - w / 10, base_y - th, white, alpha + 0.02, 1);
        draw_line(frame, width, height, tx + w / 10, base_y, tx + w / 10, base_y - th, white, alpha + 0.02, 1);
        draw_line(frame, width, height, tx - w / 10, base_y - th, tx, base_y - th - h / 5, white, alpha + 0.02, 1);
        draw_line(frame, width, height, tx, base_y - th - h / 5, tx + w / 10, base_y - th, white, alpha + 0.02, 1);
    }
    if brass.onset > 0.35 || woodwinds.onset > 0.35 {
        let flare = brass.onset.max(woodwinds.onset);
        fill_ellipse(frame, width, height, cx, base_y - h - lift, (w as f32 * 0.38) as i32, (h as f32 * 0.22) as i32, white, 0.10 + flare * 0.18);
    }
    let moon_x = (width as f32 * (0.77 + 0.015 * (t * 0.05).sin())) as i32;
    fill_ellipse(frame, width, height, moon_x, (height as f32 * 0.18) as i32, 34, 34, Color::new(160.0, 176.0, 198.0), 0.04);
}

fn draw_battlefield_floor_waves(frame: &mut [u8], width: usize, height: usize, bass: Meters, guitar: Meters, t: f32) {
    let red = Color::new(132.0, 46.0, 34.0);
    for band in 0..7 {
        let y = height as f32 * (0.64 + band as f32 * 0.045);
        let amp = 6.0 + bass.bass * 18.0 + guitar.mid * 6.0;
        let mut prev = None;
        for step in 0..100 {
            let n = step as f32 / 99.0;
            let x = n * width as f32;
            let wave = (n * 9.0 + t * (0.32 + bass.rms * 0.25) + band as f32).sin() * amp;
            let point = (x as i32, (y + wave) as i32);
            if let Some((px, py)) = prev {
                draw_line(frame, width, height, px, py, point.0, point.1, red, 0.08 + bass.rms * 0.08, 1);
            }
            prev = Some(point);
        }
    }
}

fn draw_strings_fate_arcs(frame: &mut [u8], width: usize, height: usize, strings: Meters, t: f32) {
    let color = Color::new(196.0, 184.0, 228.0);
    let cx = width as f32 * 0.50;
    let cy = height as f32 * 0.49;
    for arc in 0..7 {
        let rx = width as f32 * (0.16 + arc as f32 * 0.045 + strings.rms * 0.035);
        let ry = height as f32 * (0.07 + arc as f32 * 0.016);
        let rot = t * (2.0 + strings.mid * 1.4) + arc as f32 * 13.0;
        draw_ellipse_outline(frame, width, height, cx as i32, cy as i32, rx as i32, ry as i32, rot, color, 0.10 + strings.rms * 0.13);
    }
}

fn draw_dual_distance_traces(frame: &mut [u8], width: usize, height: usize, lead: Meters, backing: Meters, strings: Meters, t: f32) {
    let king = Color::new(220.0, 110.0, 62.0);
    let queen = Color::new(182.0, 210.0, 240.0);
    let y_base = height as f32 * 0.58;
    let gap = width as f32 * (0.18 - lead.rms.min(0.85) * 0.060);
    let left_end = width as f32 * 0.5 - gap;
    let right_end = width as f32 * 0.5 + gap;
    let mut prev_l = None;
    let mut prev_r = None;
    for step in 0..120 {
        let n = step as f32 / 119.0;
        let x_l = width as f32 * 0.07 + n * (left_end - width as f32 * 0.07);
        let x_r = width as f32 * 0.93 - n * (width as f32 * 0.93 - right_end);
        let jag = (t * 2.4 + n * 19.0).sin() * (5.0 + lead.mid * 14.0);
        let smooth = (t * 0.8 + n * 8.0).sin() * (3.0 + backing.rms * 7.0);
        let y_l = y_base + jag - n * strings.rms * 28.0;
        let y_r = y_base + smooth - n * backing.mid * 18.0;
        let p_l = (x_l as i32, y_l as i32);
        let p_r = (x_r as i32, y_r as i32);
        if let Some((px, py)) = prev_l {
            draw_line(frame, width, height, px, py, p_l.0, p_l.1, king, 0.26 + lead.rms * 0.22, 2);
        }
        if let Some((px, py)) = prev_r {
            draw_line(frame, width, height, px, py, p_r.0, p_r.1, queen, 0.22 + backing.rms * 0.20, 2);
        }
        prev_l = Some(p_l);
        prev_r = Some(p_r);
    }
}

fn draw_synth_halo_depth_haze(frame: &mut [u8], width: usize, height: usize, synth: Meters, other: Meters, t: f32) {
    let cx = width as i32 / 2;
    let cy = (height as f32 * 0.49) as i32;
    let color = Color::new(80.0, 92.0, 126.0);
    for layer in 0..5 {
        let rx = (width as f32 * (0.22 + layer as f32 * 0.08 + synth.rms * 0.05)) as i32;
        let ry = (height as f32 * (0.075 + layer as f32 * 0.035 + other.rms * 0.02)) as i32;
        let drift = ((t * 0.25 + layer as f32).sin() * 12.0) as i32;
        fill_ellipse(frame, width, height, cx + drift, cy, rx, ry, color, 0.018 + synth.rms * 0.016);
    }
}

fn draw_lead_vocal_ribbon(frame: &mut [u8], width: usize, height: usize, lead: Meters, t: f32) {
    let color = Color::new(255.0, 198.0, 218.0);
    let cx = width as f32 * 0.5;
    let top = height as f32 * 0.22;
    let bottom = height as f32 * 0.76;
    let mut prev = None;
    for step in 0..120 {
        let n = step as f32 / 119.0;
        let y = top + n * (bottom - top);
        let x = cx + (t * (1.1 + lead.mid * 1.5) + n * 10.0).sin() * width as f32 * (0.012 + lead.rms * 0.030);
        let point = (x as i32, y as i32);
        if let Some((px, py)) = prev {
            draw_line(frame, width, height, px, py, point.0, point.1, color, 0.24 + lead.rms * 0.33, 2);
        }
        prev = Some(point);
    }
}

fn draw_backing_vocal_orbits(frame: &mut [u8], width: usize, height: usize, backing: Meters, t: f32) {
    let color = Color::new(178.0, 186.0, 228.0);
    for orbit in 0..4 {
        let rx = (width as f32 * (0.07 + orbit as f32 * 0.035 + backing.rms * 0.035)) as i32;
        let ry = (height as f32 * (0.05 + orbit as f32 * 0.018 + backing.mid * 0.018)) as i32;
        draw_ellipse_outline(
            frame,
            width,
            height,
            width as i32 / 2,
            (height as f32 * 0.50) as i32,
            rx,
            ry,
            t * (18.0 + backing.high * 22.0) + orbit as f32 * 41.0,
            color,
            0.09 + backing.rms * 0.16,
        );
    }
}

fn draw_guitar_warm_trails(frame: &mut [u8], width: usize, height: usize, guitar: Meters, t: f32) {
    let color = Color::new(226.0, 132.0, 62.0);
    for row in 0..8 {
        let y = height as f32 * (0.31 + row as f32 * 0.041);
        let offset = (t * (0.28 + guitar.mid * 0.45) + row as f32).sin() * width as f32 * 0.05;
        draw_line(
            frame,
            width,
            height,
            (width as f32 * 0.12 + offset) as i32,
            y as i32,
            (width as f32 * 0.88 + offset * 0.35) as i32,
            (y + guitar.high * 12.0) as i32,
            color,
            0.08 + guitar.rms * 0.14,
            1,
        );
    }
}

fn draw_drums_and_percussion(frame: &mut [u8], width: usize, height: usize, drums: Meters, percussion: Meters, frame_index: usize, t: f32) {
    let impact = drums.onset.max(drums.rms * 0.45);
    if impact > 0.18 {
        let alpha = (impact * 0.24).min(0.20);
        draw_line(frame, width, height, width as i32 / 2, 0, width as i32 / 2, height as i32, Color::new(170.0, 138.0, 104.0), alpha, 1);
    }
    let mut rng = XorShift::new(frame_index as u64 * 97 + 17);
    let count = (percussion.onset * 18.0 + percussion.high * 10.0) as usize;
    for _ in 0..count {
        let x = (rng.next_f32() * width as f32) as i32;
        let y = (height as f32 * (0.34 + rng.next_f32() * 0.42)) as i32;
        let color = Color::new(220.0, 202.0, 168.0);
        draw_line(frame, width, height, x - 2, y, x + 2, y + (t.sin() * 2.0) as i32, color, 0.18 + percussion.high * 0.18, 1);
    }
}

fn draw_soft_stem_meter_overlay(frame: &mut [u8], width: usize, height: usize, tracks: &[StemTrack], frame_index: usize) {
    let panel_x = 14;
    let panel_y = height as i32 - 74;
    let bar_w = (width as f32 * 0.10).max(70.0) as i32;
    fill_rect(frame, width, height, panel_x - 8, panel_y - 8, (bar_w + 24) * 6, 62, Color::new(0.0, 0.0, 0.0), 0.40);
    for (index, track) in tracks.iter().take(6).enumerate() {
        let meter = scaled_meters_for(tracks, track.id, frame_index);
        let value = meter.rms.max(meter.mid).max(meter.bass).clamp(0.0, 1.0);
        let x = panel_x + index as i32 * (bar_w + 20);
        let y = panel_y + 22;
        fill_rect(frame, width, height, x, y, bar_w, 6, Color::new(20.0, 22.0, 28.0), 0.70);
        fill_rect(frame, width, height, x, y, (bar_w as f32 * value) as i32, 6, soft_lane_color(track.id), 0.65);
    }
}

fn soft_lane_color(id: &str) -> Color {
    match id {
        "lead_vocals" => Color::new(255.0, 198.0, 218.0),
        "backing_vocals" => Color::new(178.0, 186.0, 228.0),
        "drums" => Color::new(188.0, 96.0, 76.0),
        "bass" => Color::new(160.0, 92.0, 54.0),
        "guitar" => Color::new(226.0, 132.0, 62.0),
        "keyboard" => Color::new(92.0, 116.0, 140.0),
        "strings" => Color::new(196.0, 184.0, 228.0),
        _ => Color::new(148.0, 158.0, 176.0),
    }
}

fn blend_pixel(frame: &mut [u8], width: usize, height: usize, x: i32, y: i32, color: Color, alpha: f32) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 || alpha <= 0.0 {
        return;
    }
    let i = (y as usize * width + x as usize) * 3;
    frame[i] = (frame[i] as f32 + color.r * alpha).clamp(0.0, 255.0) as u8;
    frame[i + 1] = (frame[i + 1] as f32 + color.g * alpha).clamp(0.0, 255.0) as u8;
    frame[i + 2] = (frame[i + 2] as f32 + color.b * alpha).clamp(0.0, 255.0) as u8;
}

fn draw_line(frame: &mut [u8], width: usize, height: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: Color, alpha: f32, thickness: i32) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        for oy in -thickness..=thickness {
            for ox in -thickness..=thickness {
                if ox * ox + oy * oy <= thickness * thickness {
                    blend_pixel(frame, width, height, x + ox, y + oy, color, alpha);
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn fill_rect(frame: &mut [u8], width: usize, height: usize, x: i32, y: i32, w: i32, h: i32, color: Color, alpha: f32) {
    if w <= 0 || h <= 0 {
        return;
    }
    for yy in y.max(0)..(y + h).min(height as i32) {
        for xx in x.max(0)..(x + w).min(width as i32) {
            blend_pixel(frame, width, height, xx, yy, color, alpha);
        }
    }
}

fn fill_ellipse(frame: &mut [u8], width: usize, height: usize, cx: i32, cy: i32, rx: i32, ry: i32, color: Color, alpha: f32) {
    if rx <= 0 || ry <= 0 {
        return;
    }
    for y in (cy - ry).max(0)..=(cy + ry).min(height as i32 - 1) {
        for x in (cx - rx).max(0)..=(cx + rx).min(width as i32 - 1) {
            let dx = (x - cx) as f32 / rx as f32;
            let dy = (y - cy) as f32 / ry as f32;
            let d = dx * dx + dy * dy;
            if d <= 1.0 {
                blend_pixel(frame, width, height, x, y, color, alpha * (1.0 - d).max(0.0));
            }
        }
    }
}

fn draw_ellipse_outline(frame: &mut [u8], width: usize, height: usize, cx: i32, cy: i32, rx: i32, ry: i32, rotation_deg: f32, color: Color, alpha: f32) {
    let rot = rotation_deg.to_radians();
    let cos_r = rot.cos();
    let sin_r = rot.sin();
    let mut prev: Option<(i32, i32)> = None;
    for step in 0..=144 {
        let a = step as f32 / 144.0 * std::f32::consts::TAU;
        let ex = a.cos() * rx as f32;
        let ey = a.sin() * ry as f32;
        let x = cx + (ex * cos_r - ey * sin_r) as i32;
        let y = cy + (ex * sin_r + ey * cos_r) as i32;
        if let Some((px, py)) = prev {
            draw_line(frame, width, height, px, py, x, y, color, alpha, 1);
        }
        prev = Some((x, y));
    }
}

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 40) as f32) / ((1_u64 << 24) as f32)
    }
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

fn build_meter_summary(args: &Args, tracks: &[StemTrack], frame_count: usize) -> String {
    let mut stem_json = String::new();
    for (index, track) in tracks.iter().enumerate() {
        if index > 0 {
            stem_json.push(',');
        }
        stem_json.push_str(&format!(
            "{{\"stem_id\":\"{}\",\"label\":\"{}\",\"source_path\":\"{}\",\"visual_lane\":\"{}\",\"activity_scale\":{:.6}}}",
            track.id,
            track.label,
            json_escape(&track.source_path.display().to_string()),
            track.lane,
            track.activity_scale
        ));
    }
    format!(
        "{{\n  \"schema_version\":\"truevision_distant_love_meter_summary_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"distant_love_state_soft\",\n  \"master_audio\":\"{}\",\n  \"stems_dir\":\"{}\",\n  \"sample_rate\":{},\n  \"fps\":{},\n  \"duration_seconds\":{},\n  \"frame_count\":{},\n  \"stem_sources\":[{}],\n  \"boundary\":{{\"stem_directory_controls\":true,\"laser_show_preset\":false,\"soft_terror_balance\":true}}\n}}\n",
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_dir.display().to_string()),
        SAMPLE_RATE,
        args.fps,
        args.duration,
        frame_count,
        stem_json
    )
}

fn frame_state_json(tracks: &[StemTrack], frame_index: usize, time_seconds: f64) -> String {
    let mut controls = String::new();
    for (index, track) in tracks.iter().enumerate() {
        if index > 0 {
            controls.push(',');
        }
        let m = scaled_meters_for(tracks, track.id, frame_index);
        controls.push_str(&format!(
            "\"{}\":{{\"label\":\"{}\",\"visual_lane\":\"{}\",\"activity_scale\":{:.6},\"meters\":{{\"rms\":{:.6},\"onset\":{:.6},\"bass\":{:.6},\"mid\":{:.6},\"high\":{:.6}}}}}",
            track.id, track.label, track.lane, track.activity_scale, m.rms, m.onset, m.bass, m.mid, m.high
        ));
    }
    format!(
        "{{\"schema_version\":\"truevision_distant_love_frame_state_rs_v1\",\"renderer\":\"rust\",\"preset\":\"distant_love_state_soft\",\"frame_index\":{},\"time_seconds\":{:.6},\"stem_controls\":{{{}}},\"render_lanes\":{{\"lead_vocal_ribbon\":\"soft_white_pink_breath_line\",\"strings_fate_arcs\":\"graceful_slow_curves\",\"keyboard_harmonic_lattice\":\"soft_harmonic_lattice\",\"soft_terror_balance\":\"distance_signal_longing_orbit_memory\"}},\"boundary\":{{\"stem_directory_controls\":true,\"laser_show_preset\":false,\"generated_media_is_evidence\":false}}}}",
        frame_index, time_seconds, controls
    )
}

fn manifest_json(
    args: &Args,
    video_path: &PathBuf,
    state_path: &PathBuf,
    meter_path: &PathBuf,
    frame_count: usize,
    wall_seconds: f64,
) -> String {
    format!(
        "{{\n  \"schema_version\":\"truevision_distant_love_state_soft_manifest_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"distant_love_state_soft\",\n  \"run_id\":\"{}\",\n  \"source\":{{\"master_audio\":\"{}\",\"stems_dir\":\"{}\"}},\n  \"output\":{{\"mp4\":\"{}\",\"frame_state_jsonl\":\"{}\",\"stem_meter_summary_json\":\"{}\",\"width\":{},\"height\":{},\"fps\":{},\"duration_seconds\":{},\"frame_count\":{},\"encoder\":\"{}\",\"wall_seconds\":{:.6}}},\n  \"visual_contract\":{{\"goal\":\"music state made visible\",\"intensity\":\"restrained_emotional_soft_terror\",\"forbidden\":[\"laser_show_preset\",\"over_bloom\",\"beam_cage\",\"foreground_wipeouts\",\"people\"]}},\n  \"boundary\":{{\"stem_directory_controls\":true,\"laser_show_preset\":false,\"soft_terror_balance\":true,\"master_audio_drives_final_sound\":true,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_dir.display().to_string()),
        json_escape(&video_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        json_escape(&meter_path.display().to_string()),
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
        "{{\n  \"schema_version\":\"truevision_distant_love_state_soft_receipt_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"distant_love_state_soft\",\n  \"run_id\":\"{}\",\n  \"status\":\"complete\",\n  \"output_mp4\":\"{}\",\n  \"manifest_json\":\"{}\",\n  \"frame_state_jsonl\":\"{}\",\n  \"elapsed_seconds\":{:.6},\n  \"boundary\":{{\"stem_directory_controls\":true,\"laser_show_preset\":false,\"soft_terror_balance\":true,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        wall_seconds
    )
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
