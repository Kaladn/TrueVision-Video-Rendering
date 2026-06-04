use std::env;
use std::fs::{create_dir_all, read_dir, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const STEMS: [(&str, &str, &str); 8] = [
    ("lead_vocals", "Lead Vocals", "main_gold_ember_thread"),
    (
        "backing_vocals",
        "Backing Vocals",
        "answering_rose_ember_thread",
    ),
    ("drums", "Drums", "pressure_hits"),
    ("bass", "Bass", "flats_river_depth_pulse"),
    ("keyboard", "Keyboard", "memory_lattice"),
    ("percussion", "Percussion", "ash_spark_nerves"),
    ("synth", "Synth", "phoenix_heat_veil"),
    ("other", "Other", "steel_air_texture"),
];
const PHASE_NAMES: [&str; 6] = [
    "lineart_damage_state",
    "witness_expansion",
    "dual_descent",
    "impact_transform",
    "regrowth_wave",
    "healed_forest_state",
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

#[derive(Clone, Copy)]
struct PhoenixPhase {
    index: usize,
    name: &'static str,
    progress: f32,
    lineart_ratio: f32,
    destructive_fire_ratio: f32,
    witness_expansion_ratio: f32,
    dual_vector_pressure: f32,
    impact_pressure: f32,
    healing_color_ratio: f32,
    regrowth_ratio: f32,
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
        render_frame(
            &args,
            &tracks,
            frame_index,
            frame_count,
            time_seconds as f32,
            &mut frame,
        );
        stdin
            .write_all(&frame)
            .map_err(|e| format!("ffmpeg write failed: {e}"))?;
        if frame_index % args.state_log_every == 0 {
            writeln!(
                state_file,
                "{}",
                frame_state_json(&tracks, frame_index, frame_count, time_seconds)
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
                &tracks,
                &video_path,
                &state_path,
                &meter_path,
                frame_count,
                wall_seconds,
            )
            .as_bytes(),
        )
        .map_err(|e| format!("manifest write failed: {e}"))?;
    File::create(&receipt_path)
        .map_err(|e| format!("receipt open failed: {e}"))?
        .write_all(
            receipt_json(
                &args,
                &video_path,
                &manifest_path,
                &state_path,
                wall_seconds,
            )
            .as_bytes(),
        )
        .map_err(|e| format!("receipt write failed: {e}"))?;

    println!(
        "{{\"renderer\":\"rust\",\"preset\":\"phoenix_from_the_flats_state_v0\",\"video_path\":\"{}\",\"manifest_path\":\"{}\",\"frame_count\":{},\"wall_seconds\":{:.3}}}",
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        frame_count,
        wall_seconds
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        output_root: PathBuf::from("outputs/phoenix_from_the_flats_state"),
        run_id: "phoenix_from_the_flats_30s".to_string(),
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
            "--duration" => {
                args.duration = value.parse().map_err(|_| "bad duration".to_string())?
            }
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
        let mut meters = compute_meters(&samples, SAMPLE_RATE, fps, frame_count);
        smooth_meters(&mut meters);
        let activity_scale = lane_presence_floor(id, activity_scale(&samples));
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

fn find_stem(stems_dir: &Path, label: &str) -> Result<PathBuf, String> {
    let needle = label.to_ascii_lowercase();
    let compact_needle = needle.replace(' ', "");
    let mut candidates = Vec::new();
    for entry in read_dir(stems_dir).map_err(|e| format!("read stems dir failed: {e}"))? {
        let path = entry
            .map_err(|e| format!("read stem entry failed: {e}"))?
            .path();
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

fn activity_scale(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut peak = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for sample in samples {
        let abs = sample.abs();
        peak = peak.max(abs);
        sum_sq += sample * sample;
    }
    let rms = (sum_sq / samples.len() as f32).sqrt();
    ((rms * 14.0).max(peak * 1.7)).clamp(0.05, 1.0)
}

fn lane_presence_floor(id: &str, scale: f32) -> f32 {
    let floor = match id {
        "lead_vocals" => 0.42,
        "backing_vocals" => 0.34,
        "keyboard" => 0.28,
        "percussion" => 0.20,
        "synth" => 0.30,
        "other" => 0.24,
        _ => 0.16,
    };
    scale.max(floor).min(1.0)
}

fn compute_meters(
    samples: &[f32],
    sample_rate: usize,
    fps: usize,
    frame_count: usize,
) -> Vec<Meters> {
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

fn smooth_meters(meters: &mut [Meters]) {
    if meters.len() < 3 {
        return;
    }
    let original = meters.to_vec();
    for index in 0..meters.len() {
        let lo = index.saturating_sub(2);
        let hi = (index + 2).min(meters.len() - 1);
        let mut out = Meters::default();
        let count = (hi - lo + 1) as f32;
        for meter in &original[lo..=hi] {
            out.rms += meter.rms;
            out.onset += meter.onset;
            out.bass += meter.bass;
            out.mid += meter.mid;
            out.high += meter.high;
        }
        meters[index] = Meters {
            rms: out.rms / count,
            onset: out.onset / count,
            bass: out.bass / count,
            mid: out.mid / count,
            high: out.high / count,
        };
    }
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

fn render_frame(
    args: &Args,
    tracks: &[StemTrack],
    frame_index: usize,
    frame_count: usize,
    t: f32,
    frame: &mut [u8],
) {
    let phase = phoenix_phase(frame_index, frame_count);
    let lead = scaled_meters_for(tracks, "lead_vocals", frame_index);
    let backing = scaled_meters_for(tracks, "backing_vocals", frame_index);
    let drums = scaled_meters_for(tracks, "drums", frame_index);
    let bass = scaled_meters_for(tracks, "bass", frame_index);
    let keyboard = scaled_meters_for(tracks, "keyboard", frame_index);
    let percussion = scaled_meters_for(tracks, "percussion", frame_index);
    let synth = scaled_meters_for(tracks, "synth", frame_index);
    let other = scaled_meters_for(tracks, "other", frame_index);

    fill_flats_background(frame, args.width, args.height, synth, bass, t);
    draw_flats_silhouette(frame, args.width, args.height, keyboard, other, t);
    draw_river_depth_pulse(frame, args.width, args.height, bass, keyboard, t);
    draw_lineart_world_state(
        frame,
        args.width,
        args.height,
        phase,
        keyboard,
        other,
        t,
    );
    draw_memory_lattice(frame, args.width, args.height, keyboard, t);
    draw_phoenix_heat_veil(frame, args.width, args.height, synth, lead, backing, t);
    draw_dual_ember_threads(
        frame,
        args.width,
        args.height,
        lead,
        backing,
        drums,
        synth,
        t,
    );
    draw_pressure_hits(frame, args.width, args.height, drums, bass, frame_index, t);
    draw_ash_sparks(frame, args.width, args.height, percussion, frame_index, t);
    draw_steel_air_texture(frame, args.width, args.height, other, synth, t);
    draw_dual_descent_vectors(
        frame,
        args.width,
        args.height,
        phase,
        lead,
        backing,
        drums,
        t,
    );
    draw_regrowth_wave(
        frame,
        args.width,
        args.height,
        phase,
        bass,
        keyboard,
        synth,
        frame_index,
        t,
    );
    apply_phoenix_phase_grade(frame, args.width, args.height, phase);
    draw_minimal_stem_meter(frame, args.width, args.height, tracks, frame_index);
}

fn phoenix_phase(frame_index: usize, frame_count: usize) -> PhoenixPhase {
    let progress = if frame_count <= 1 {
        1.0
    } else {
        frame_index as f32 / (frame_count - 1) as f32
    }
    .clamp(0.0, 1.0);
    let (index, name) = if progress < 0.25 {
        (0, "lineart_damage_state")
    } else if progress < 0.45 {
        (1, "witness_expansion")
    } else if progress < 0.60 {
        (2, "dual_descent")
    } else if progress < 0.78 {
        (3, "impact_transform")
    } else if progress < 0.93 {
        (4, "regrowth_wave")
    } else {
        (5, "healed_forest_state")
    };
    PhoenixPhase {
        index,
        name,
        progress,
        lineart_ratio: (1.0 - smoothstep(0.18, 0.64, progress)).clamp(0.0, 1.0),
        destructive_fire_ratio: (smoothstep(0.18, 0.62, progress)
            * (1.0 - smoothstep(0.76, 0.96, progress)))
        .clamp(0.0, 1.0),
        witness_expansion_ratio: smoothstep(0.22, 0.48, progress),
        dual_vector_pressure: (smoothstep(0.44, 0.58, progress)
            * (1.0 - smoothstep(0.70, 0.88, progress)))
        .clamp(0.0, 1.0),
        impact_pressure: (smoothstep(0.58, 0.68, progress)
            * (1.0 - smoothstep(0.74, 0.86, progress)))
        .clamp(0.0, 1.0),
        healing_color_ratio: smoothstep(0.70, 0.98, progress),
        regrowth_ratio: smoothstep(0.78, 0.98, progress),
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    let x = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn fill_flats_background(
    frame: &mut [u8],
    width: usize,
    height: usize,
    synth: Meters,
    bass: Meters,
    t: f32,
) {
    for y in 0..height {
        let yy = y as f32 / height.max(1) as f32;
        let pulse = 0.5 + 0.5 * (t * 0.22).sin();
        let river = (yy - 0.62).max(0.0) * 1.7;
        let r =
            (0.010 + yy * 0.018 + bass.bass * 0.020 * river + synth.rms * 0.010 * pulse) * 255.0;
        let g = (0.012 + yy * 0.010 + synth.mid * 0.008) * 255.0;
        let b = (0.026 + (1.0 - yy) * (0.038 + synth.high * 0.018) + river * 0.010) * 255.0;
        for x in 0..width {
            let i = (y * width + x) * 3;
            frame[i] = r as u8;
            frame[i + 1] = g as u8;
            frame[i + 2] = b as u8;
        }
    }
}

fn draw_flats_silhouette(
    frame: &mut [u8],
    width: usize,
    height: usize,
    keyboard: Meters,
    other: Meters,
    t: f32,
) {
    let horizon = (height as f32 * 0.56) as i32;
    let dark = Color::new(3.0, 5.0, 7.0);
    fill_rect(
        frame,
        width,
        height,
        0,
        horizon,
        width as i32,
        10,
        dark,
        0.84,
    );
    for index in 0..18 {
        let n = index as f32 / 17.0;
        let x = (width as f32 * n) as i32;
        let h = (height as f32 * (0.07 + 0.08 * ((index * 37 % 11) as f32 / 10.0))) as i32;
        let wobble = ((t * 0.08 + index as f32).sin() * other.rms * 2.0) as i32;
        fill_rect(
            frame,
            width,
            height,
            x - 8,
            horizon - h + wobble,
            16,
            h,
            dark,
            0.72,
        );
    }
    let line = Color::new(
        66.0,
        72.0 + keyboard.mid * 45.0,
        76.0 + keyboard.high * 38.0,
    );
    for bridge in 0..3 {
        let y = horizon - 20 - bridge * 23;
        draw_line(
            frame,
            width,
            height,
            0,
            y,
            width as i32,
            y - 8,
            line,
            0.08 + keyboard.rms * 0.10,
            1,
        );
    }
}

fn draw_river_depth_pulse(
    frame: &mut [u8],
    width: usize,
    height: usize,
    bass: Meters,
    keyboard: Meters,
    t: f32,
) {
    let water_top = height as f32 * 0.62;
    let color = Color::new(122.0 + bass.bass * 36.0, 74.0 + keyboard.mid * 28.0, 44.0);
    for band in 0..12 {
        let n_band = band as f32 / 11.0;
        let y = water_top + n_band * height as f32 * 0.32;
        let amp = 2.0 + bass.rms * 11.0 + keyboard.rms * 5.0;
        let mut prev = None;
        for step in 0..140 {
            let n = step as f32 / 139.0;
            let x = n * width as f32;
            let wave = (n * 12.0 + t * (0.30 + bass.bass * 0.28) + band as f32 * 0.7).sin() * amp;
            let point = (x as i32, (y + wave) as i32);
            if let Some((px, py)) = prev {
                draw_line(
                    frame,
                    width,
                    height,
                    px,
                    py,
                    point.0,
                    point.1,
                    color,
                    0.035 + bass.rms * 0.080,
                    1,
                );
            }
            prev = Some(point);
        }
    }
}

fn draw_lineart_world_state(
    frame: &mut [u8],
    width: usize,
    height: usize,
    phase: PhoenixPhase,
    keyboard: Meters,
    other: Meters,
    t: f32,
) {
    let horizon = height as f32 * 0.58;
    let line_alpha =
        (0.10 + phase.lineart_ratio * 0.24 + phase.healing_color_ratio * 0.13 + keyboard.rms * 0.06)
            .clamp(0.08, 0.48);
    let city_color = Color::new(
        112.0 + phase.healing_color_ratio * 34.0,
        126.0 + phase.healing_color_ratio * 70.0,
        132.0 + phase.healing_color_ratio * 42.0,
    );
    for index in 0..22 {
        let n = index as f32 / 21.0;
        let x = width as f32 * (0.045 + n * 0.91);
        let w = width as f32 * (0.012 + ((index * 17 % 7) as f32) * 0.003);
        let h = height as f32 * (0.055 + ((index * 29 % 11) as f32) * 0.010);
        let y0 = horizon - h;
        let lean = ((t * 0.10 + index as f32 * 0.61).sin() * other.rms * 5.0) as i32;
        draw_line(
            frame,
            width,
            height,
            (x - w) as i32 + lean,
            horizon as i32,
            (x - w) as i32,
            y0 as i32,
            city_color,
            line_alpha,
            1,
        );
        draw_line(
            frame,
            width,
            height,
            (x + w) as i32 + lean,
            horizon as i32,
            (x + w) as i32,
            y0 as i32,
            city_color,
            line_alpha,
            1,
        );
        draw_line(
            frame,
            width,
            height,
            (x - w) as i32,
            y0 as i32,
            (x + w) as i32,
            (y0 + ((index * 13 % 5) as f32 - 2.0)) as i32,
            city_color,
            line_alpha,
            1,
        );
    }

    let scorch = phase.destructive_fire_ratio * (1.0 - phase.regrowth_ratio * 0.80);
    let branch_color = Color::new(
        86.0 + phase.healing_color_ratio * 42.0 + scorch * 86.0,
        88.0 + phase.healing_color_ratio * 126.0 + scorch * 26.0,
        84.0 + phase.healing_color_ratio * 60.0,
    );
    for index in 0..34 {
        let n = index as f32 / 33.0;
        let x = width as f32 * (0.03 + n * 0.94);
        let base_y = height as f32 * (0.76 + ((index * 19 % 9) as f32) * 0.017);
        let height_scale = height as f32
            * (0.055 + phase.regrowth_ratio * 0.105 + ((index * 31 % 5) as f32) * 0.010);
        let sway = (t * (0.11 + phase.regrowth_ratio * 0.20) + n * 7.0).sin()
            * (2.0 + keyboard.mid * 8.0);
        let top_x = x + sway;
        let top_y = base_y - height_scale;
        draw_line(
            frame,
            width,
            height,
            x as i32,
            base_y as i32,
            top_x as i32,
            top_y as i32,
            branch_color,
            0.09 + phase.lineart_ratio * 0.10 + phase.regrowth_ratio * 0.18,
            1,
        );
        let fork = 8.0 + phase.regrowth_ratio * 14.0;
        draw_line(
            frame,
            width,
            height,
            top_x as i32,
            top_y as i32,
            (top_x - fork) as i32,
            (top_y + fork * 0.45) as i32,
            branch_color,
            0.06 + phase.regrowth_ratio * 0.14,
            1,
        );
        draw_line(
            frame,
            width,
            height,
            top_x as i32,
            top_y as i32,
            (top_x + fork * 0.8) as i32,
            (top_y + fork * 0.38) as i32,
            branch_color,
            0.06 + phase.regrowth_ratio * 0.14,
            1,
        );
    }

    if phase.regrowth_ratio > 0.72 {
        let canopy_color = Color::new(56.0, 186.0, 116.0);
        for row in 0..3 {
            let y = height as f32 * (0.66 + row as f32 * 0.045);
            let mut prev = None;
            for step in 0..120 {
                let n = step as f32 / 119.0;
                let x = n * width as f32;
                let wave = (n * 22.0 + t * 0.18 + row as f32).sin()
                    * height as f32
                    * 0.010
                    * phase.regrowth_ratio;
                let point = (x as i32, (y + wave) as i32);
                if let Some((px, py)) = prev {
                    draw_line(
                        frame,
                        width,
                        height,
                        px,
                        py,
                        point.0,
                        point.1,
                        canopy_color,
                        0.030 + phase.regrowth_ratio * 0.070,
                        1,
                    );
                }
                prev = Some(point);
            }
        }
    }
}

fn draw_memory_lattice(frame: &mut [u8], width: usize, height: usize, keyboard: Meters, t: f32) {
    let color = Color::new(96.0, 116.0, 126.0);
    for index in 0..10 {
        let x = (width as f32 * (0.12 + index as f32 * 0.085)) as i32;
        let lean = ((t * 0.16 + index as f32).sin() * keyboard.mid * 10.0) as i32;
        draw_line(
            frame,
            width,
            height,
            x + lean,
            (height as f32 * 0.20) as i32,
            x - lean,
            (height as f32 * 0.63) as i32,
            color,
            0.025 + keyboard.rms * 0.075,
            1,
        );
    }
    for row in 0..7 {
        let y = height as f32 * (0.24 + row as f32 * 0.055);
        draw_line(
            frame,
            width,
            height,
            (width as f32 * 0.10) as i32,
            y as i32,
            (width as f32 * 0.90) as i32,
            (y + (t * 0.19 + row as f32).cos() * keyboard.high * 7.0) as i32,
            color,
            0.022 + keyboard.mid * 0.060,
            1,
        );
    }
}

fn draw_phoenix_heat_veil(
    frame: &mut [u8],
    width: usize,
    height: usize,
    synth: Meters,
    lead: Meters,
    backing: Meters,
    t: f32,
) {
    let cx = width as i32 / 2;
    let cy = (height as f32 * 0.48) as i32;
    let warmth = (lead.rms + backing.rms + synth.mid).min(1.6);
    for layer in 0..7 {
        let rx = (width as f32 * (0.12 + layer as f32 * 0.045 + synth.rms * 0.045)) as i32;
        let ry = (height as f32 * (0.040 + layer as f32 * 0.020 + synth.high * 0.020)) as i32;
        let rot = t * (5.0 + synth.mid * 9.0) + layer as f32 * 19.0;
        let drift = ((t * 0.23 + layer as f32).sin() * 18.0 * synth.rms) as i32;
        draw_ellipse_outline(
            frame,
            width,
            height,
            cx + drift,
            cy,
            rx,
            ry,
            rot,
            Color::new(210.0 + warmth * 20.0, 84.0 + synth.rms * 54.0, 36.0),
            0.070 + synth.rms * 0.105,
        );
    }
}

fn draw_dual_ember_threads(
    frame: &mut [u8],
    width: usize,
    height: usize,
    lead: Meters,
    backing: Meters,
    drums: Meters,
    synth: Meters,
    t: f32,
) {
    let y_base = height as f32 * 0.50;
    let pressure = (lead.rms + backing.rms + drums.onset * 0.45).min(1.5);
    let gap = width as f32 * (0.16 - pressure * 0.035).max(0.075);
    let left_end = width as f32 * 0.5 - gap;
    let right_end = width as f32 * 0.5 + gap;
    let mut prev_l = None;
    let mut prev_r = None;
    for step in 0..150 {
        let n = step as f32 / 149.0;
        let x_l = width as f32 * 0.06 + n * (left_end - width as f32 * 0.06);
        let x_r = width as f32 * 0.94 - n * (width as f32 * 0.94 - right_end);
        let left_wave = (t * (0.75 + lead.mid * 1.10) + n * 12.0).sin() * (7.0 + lead.rms * 20.0);
        let right_wave =
            (t * (0.70 + backing.mid * 1.05) + n * 10.5 + 1.4).sin() * (6.0 + backing.rms * 18.0);
        let lift = n * synth.rms * 34.0;
        let p_l = (x_l as i32, (y_base + left_wave - lift) as i32);
        let p_r = (x_r as i32, (y_base + right_wave - lift * 0.72) as i32);
        if let Some((px, py)) = prev_l {
            draw_line(
                frame,
                width,
                height,
                px,
                py,
                p_l.0,
                p_l.1,
                Color::new(255.0, 152.0, 54.0),
                0.32 + lead.rms * 0.38,
                2,
            );
        }
        if let Some((px, py)) = prev_r {
            draw_line(
                frame,
                width,
                height,
                px,
                py,
                p_r.0,
                p_r.1,
                Color::new(244.0, 88.0, 112.0),
                0.30 + backing.rms * 0.34,
                2,
            );
        }
        prev_l = Some(p_l);
        prev_r = Some(p_r);
    }
}

fn draw_pressure_hits(
    frame: &mut [u8],
    width: usize,
    height: usize,
    drums: Meters,
    bass: Meters,
    frame_index: usize,
    t: f32,
) {
    let impact = drums.onset.max(drums.rms * 0.34).max(bass.onset * 0.32);
    if impact < 0.12 {
        return;
    }
    let cx = width as i32 / 2;
    let cy = (height as f32 * (0.50 + (t * 0.12).sin() * 0.012)) as i32;
    let color = Color::new(232.0, 142.0, 78.0);
    for ring in 0..3 {
        let r = (width as f32 * (0.035 + ring as f32 * 0.026 + impact * 0.055)) as i32;
        draw_ellipse_outline(
            frame,
            width,
            height,
            cx,
            cy,
            r,
            (r as f32 * 0.42) as i32,
            0.0,
            color,
            0.05 + impact * 0.075,
        );
    }
    if frame_index % 3 == 0 {
        draw_line(
            frame,
            width,
            height,
            cx,
            cy - 70,
            cx,
            cy + 88,
            color,
            0.06 + impact * 0.12,
            1,
        );
    }
}

fn draw_ash_sparks(
    frame: &mut [u8],
    width: usize,
    height: usize,
    percussion: Meters,
    frame_index: usize,
    t: f32,
) {
    let mut rng = XorShift::new(frame_index as u64 * 101 + 71);
    let count = (percussion.onset * 22.0 + percussion.high * 11.0) as usize;
    for _ in 0..count {
        let x = (rng.next_f32() * width as f32) as i32;
        let y = (height as f32 * (0.20 + rng.next_f32() * 0.56)) as i32;
        let drift = ((t * 1.4 + rng.next_f32() * 5.0).sin() * 5.0) as i32;
        let color = Color::new(232.0, 168.0, 92.0);
        draw_line(
            frame,
            width,
            height,
            x,
            y,
            x + drift,
            y - 2,
            color,
            0.12 + percussion.high * 0.18,
            1,
        );
    }
}

fn draw_steel_air_texture(
    frame: &mut [u8],
    width: usize,
    height: usize,
    other: Meters,
    synth: Meters,
    t: f32,
) {
    let color = Color::new(74.0, 96.0, 110.0);
    for line in 0..22 {
        let n = line as f32 / 21.0;
        let x = (width as f32 * n + (t * 0.11 + line as f32).sin() * other.rms * 18.0) as i32;
        draw_line(
            frame,
            width,
            height,
            x,
            (height as f32 * 0.10) as i32,
            x + ((t * 0.17 + n).cos() * 10.0) as i32,
            (height as f32 * 0.76) as i32,
            color,
            0.012 + other.mid * 0.028 + synth.high * 0.010,
            1,
        );
    }
}

fn draw_dual_descent_vectors(
    frame: &mut [u8],
    width: usize,
    height: usize,
    phase: PhoenixPhase,
    lead: Meters,
    backing: Meters,
    drums: Meters,
    t: f32,
) {
    let pressure = (phase.dual_vector_pressure
        * (0.58 + lead.rms * 0.28 + backing.rms * 0.24 + drums.onset * 0.22))
    .clamp(0.0, 1.0);
    if pressure <= 0.01 {
        return;
    }
    let center_x = width as f32 * 0.5;
    let center_y = height as f32 * (0.50 + 0.018 * (t * 0.18).sin());
    let top_y = height as f32 * 0.08;
    let left_x = width as f32 * (0.18 + 0.025 * (t * 0.31).sin());
    let right_x = width as f32 * (0.82 + 0.025 * (t * 0.29).cos());
    let descent = smoothstep(0.44, 0.68, phase.progress);
    let left_end = (
        lerp(left_x, center_x - width as f32 * 0.025, descent),
        lerp(top_y, center_y, descent),
    );
    let right_end = (
        lerp(right_x, center_x + width as f32 * 0.025, descent),
        lerp(top_y, center_y, descent),
    );
    draw_line(
        frame,
        width,
        height,
        left_x as i32,
        top_y as i32,
        left_end.0 as i32,
        left_end.1 as i32,
        Color::new(255.0, 176.0, 72.0),
        0.12 + pressure * 0.34,
        2,
    );
    draw_line(
        frame,
        width,
        height,
        right_x as i32,
        top_y as i32,
        right_end.0 as i32,
        right_end.1 as i32,
        Color::new(246.0, 96.0, 132.0),
        0.10 + pressure * 0.30,
        2,
    );
    if phase.impact_pressure > 0.0 {
        for ring in 0..4 {
            let r = (width as f32
                * (0.028 + ring as f32 * 0.020 + phase.impact_pressure * 0.070))
                as i32;
            let color = Color::new(
                255.0,
                116.0 + phase.healing_color_ratio * 118.0,
                48.0 + phase.healing_color_ratio * 130.0,
            );
            draw_ellipse_outline(
                frame,
                width,
                height,
                center_x as i32,
                center_y as i32,
                r,
                (r as f32 * 0.56) as i32,
                t * 7.0 + ring as f32 * 21.0,
                color,
                0.07 + phase.impact_pressure * 0.13,
            );
        }
    }
}

fn draw_regrowth_wave(
    frame: &mut [u8],
    width: usize,
    height: usize,
    phase: PhoenixPhase,
    bass: Meters,
    keyboard: Meters,
    synth: Meters,
    frame_index: usize,
    t: f32,
) {
    if phase.regrowth_ratio <= 0.01 {
        return;
    }
    let mut rng = XorShift::new(frame_index as u64 * 1709 + 4307);
    let water_top = height as f32 * 0.62;
    let crest_y = lerp(height as f32 * 0.92, water_top - height as f32 * 0.13, phase.regrowth_ratio);
    let wave_color = Color::new(
        54.0 + phase.healing_color_ratio * 74.0,
        136.0 + phase.healing_color_ratio * 94.0,
        92.0 + synth.mid * 46.0,
    );
    for branch in 0..26 {
        let n = branch as f32 / 25.0;
        let x = width as f32 * n;
        let local = ((t * 0.16 + n * 5.0).sin() * 0.5 + 0.5) * keyboard.rms;
        let stem_h = height as f32 * (0.035 + phase.regrowth_ratio * 0.16 + local * 0.045);
        let y0 = crest_y + rng.next_f32() * height as f32 * 0.08;
        let y1 = y0 - stem_h;
        let sway = (t * (0.25 + bass.bass * 0.22) + n * 8.0).sin() * (3.0 + synth.rms * 9.0);
        draw_line(
            frame,
            width,
            height,
            x as i32,
            y0 as i32,
            (x + sway) as i32,
            y1 as i32,
            wave_color,
            0.028 + phase.regrowth_ratio * 0.15,
            1,
        );
        if phase.healing_color_ratio > 0.45 {
            let leaf_x = x + sway + (rng.next_f32() - 0.5) * 18.0;
            let leaf_y = y1 + (rng.next_f32() - 0.4) * 16.0;
            draw_ellipse_outline(
                frame,
                width,
                height,
                leaf_x as i32,
                leaf_y as i32,
                (3.0 + phase.healing_color_ratio * 5.0) as i32,
                (2.0 + phase.healing_color_ratio * 3.0) as i32,
                rng.next_f32() * 180.0,
                Color::new(70.0, 190.0 + keyboard.mid * 36.0, 116.0 + synth.high * 24.0),
                0.035 + phase.healing_color_ratio * 0.13,
            );
        }
    }
    for band in 0..5 {
        let y = water_top + band as f32 * height as f32 * 0.036;
        draw_line(
            frame,
            width,
            height,
            (width as f32 * 0.08) as i32,
            (y + (t * 0.21 + band as f32).sin() * 7.0 * phase.regrowth_ratio) as i32,
            (width as f32 * 0.92) as i32,
            (y + (t * 0.18 + band as f32).cos() * 6.0 * phase.regrowth_ratio) as i32,
            Color::new(54.0, 164.0 + phase.healing_color_ratio * 54.0, 172.0),
            0.018 + phase.regrowth_ratio * 0.060,
            1,
        );
    }
}

fn apply_phoenix_phase_grade(
    frame: &mut [u8],
    width: usize,
    height: usize,
    phase: PhoenixPhase,
) {
    for y in 0..height {
        let yy = y as f32 / height.max(1) as f32;
        let ground = smoothstep(0.48, 1.0, yy);
        let sky = 1.0 - ground;
        for x in 0..width {
            let i = (y * width + x) * 3;
            let r = frame[i] as f32;
            let g = frame[i + 1] as f32;
            let b = frame[i + 2] as f32;
            let gray = r * 0.299 + g * 0.587 + b * 0.114;
            let ink = phase.lineart_ratio;
            let mut nr = lerp(r, gray * 0.82 + 5.0, ink * 0.86);
            let mut ng = lerp(g, gray * 0.84 + 6.0, ink * 0.86);
            let mut nb = lerp(b, gray * 0.90 + 10.0, ink * 0.86);
            let fire = phase.destructive_fire_ratio * (0.45 + sky * 0.55);
            nr += fire * 44.0 + phase.impact_pressure * 55.0;
            ng += fire * 14.0 + phase.impact_pressure * 24.0;
            nb -= fire * 16.0;
            let heal = phase.healing_color_ratio;
            nr += heal * (18.0 + ground * 16.0);
            ng += heal * (22.0 + ground * 64.0 + phase.regrowth_ratio * 42.0);
            nb += heal * (24.0 + sky * 36.0);
            frame[i] = nr.clamp(0.0, 255.0) as u8;
            frame[i + 1] = ng.clamp(0.0, 255.0) as u8;
            frame[i + 2] = nb.clamp(0.0, 255.0) as u8;
        }
    }
}

fn lerp(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount.clamp(0.0, 1.0)
}

fn draw_minimal_stem_meter(
    frame: &mut [u8],
    width: usize,
    height: usize,
    tracks: &[StemTrack],
    frame_index: usize,
) {
    let panel_h = 28_i32;
    let top = height as i32 - panel_h - 10;
    fill_rect(
        frame,
        width,
        height,
        14,
        top - 4,
        width as i32 - 28,
        panel_h + 8,
        Color::new(0.0, 0.0, 0.0),
        0.28,
    );
    let bar_w = ((width as i32 - 64) / STEMS.len() as i32).max(24);
    for (index, track) in tracks.iter().enumerate() {
        let meter = scaled_meters_for(tracks, track.id, frame_index);
        let value = meter.rms.max(meter.mid).max(meter.bass).clamp(0.0, 1.0);
        let x = 28 + index as i32 * bar_w;
        fill_rect(
            frame,
            width,
            height,
            x,
            top + 12,
            bar_w - 8,
            5,
            Color::new(18.0, 20.0, 24.0),
            0.68,
        );
        fill_rect(
            frame,
            width,
            height,
            x,
            top + 12,
            ((bar_w - 8) as f32 * value) as i32,
            5,
            lane_color(track.id),
            0.72,
        );
    }
}

fn lane_color(id: &str) -> Color {
    match id {
        "lead_vocals" => Color::new(255.0, 152.0, 54.0),
        "backing_vocals" => Color::new(244.0, 88.0, 112.0),
        "drums" => Color::new(210.0, 76.0, 60.0),
        "bass" => Color::new(150.0, 72.0, 42.0),
        "keyboard" => Color::new(94.0, 126.0, 148.0),
        "percussion" => Color::new(236.0, 184.0, 92.0),
        "synth" => Color::new(204.0, 96.0, 58.0),
        _ => Color::new(132.0, 148.0, 160.0),
    }
}

fn blend_pixel(
    frame: &mut [u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    color: Color,
    alpha: f32,
) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 || alpha <= 0.0 {
        return;
    }
    let i = (y as usize * width + x as usize) * 3;
    frame[i] = (frame[i] as f32 + color.r * alpha).clamp(0.0, 255.0) as u8;
    frame[i + 1] = (frame[i + 1] as f32 + color.g * alpha).clamp(0.0, 255.0) as u8;
    frame[i + 2] = (frame[i + 2] as f32 + color.b * alpha).clamp(0.0, 255.0) as u8;
}

fn draw_line(
    frame: &mut [u8],
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Color,
    alpha: f32,
    thickness: i32,
) {
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

fn fill_rect(
    frame: &mut [u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    color: Color,
    alpha: f32,
) {
    let x0 = x.max(0).min(width as i32);
    let y0 = y.max(0).min(height as i32);
    let x1 = (x + w).max(0).min(width as i32);
    let y1 = (y + h).max(0).min(height as i32);
    for py in y0..y1 {
        for px in x0..x1 {
            blend_pixel(frame, width, height, px, py, color, alpha);
        }
    }
}

fn draw_ellipse_outline(
    frame: &mut [u8],
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    rotation_deg: f32,
    color: Color,
    alpha: f32,
) {
    let mut prev = None;
    let rot = rotation_deg.to_radians();
    let cos_r = rot.cos();
    let sin_r = rot.sin();
    for step in 0..96 {
        let a = step as f32 / 95.0 * std::f32::consts::TAU;
        let x = a.cos() * rx as f32;
        let y = a.sin() * ry as f32;
        let px = cx as f32 + x * cos_r - y * sin_r;
        let py = cy as f32 + x * sin_r + y * cos_r;
        let point = (px as i32, py as i32);
        if let Some((lx, ly)) = prev {
            draw_line(
                frame, width, height, lx, ly, point.0, point.1, color, alpha, 1,
            );
        }
        prev = Some(point);
    }
}

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

fn video_encode_args(args: &Args, video_path: &PathBuf) -> Vec<String> {
    vec![
        "-y".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pixel_format".to_string(),
        "rgb24".to_string(),
        "-video_size".to_string(),
        format!("{}x{}", args.width, args.height),
        "-framerate".to_string(),
        args.fps.to_string(),
        "-i".to_string(),
        "-".to_string(),
        "-an".to_string(),
        "-c:v".to_string(),
        args.video_encoder.clone(),
        "-b:v".to_string(),
        args.bitrate.clone(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        video_path.display().to_string(),
    ]
}

fn mux_audio(
    video_path: &PathBuf,
    audio_path: &PathBuf,
    output_path: &PathBuf,
    duration: f64,
) -> Result<(), String> {
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
        .map_err(|e| format!("ffmpeg mux failed: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg mux failed with status {status}"));
    }
    Ok(())
}

fn build_meter_summary(args: &Args, tracks: &[StemTrack], frame_count: usize) -> String {
    let mut sources = Vec::new();
    for track in tracks {
        sources.push(format!(
            "{{\"id\":\"{}\",\"label\":\"{}\",\"lane\":\"{}\",\"source_path\":\"{}\",\"activity_scale\":{:.6}}}",
            json_escape(track.id),
            json_escape(track.label),
            json_escape(track.lane),
            json_escape(&track.source_path.display().to_string()),
            track.activity_scale
        ));
    }
    format!(
        "{{\n  \"schema_version\":\"truevision_phoenix_flats_stem_meter_summary_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"phoenix_from_the_flats_state_v0\",\n  \"master_audio\":\"{}\",\n  \"stems_dir\":\"{}\",\n  \"fps\":{},\n  \"duration_seconds\":{},\n  \"frame_count\":{},\n  \"stem_sources\":[{}],\n  \"boundary\":{{\"python_render_loop\":false,\"external_visual_assets_used\":false,\"openai_generation_used\":false,\"stems_drive_visual_lanes\":true,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_dir.display().to_string()),
        args.fps,
        args.duration,
        frame_count,
        sources.join(",")
    )
}

fn frame_state_json(
    tracks: &[StemTrack],
    frame_index: usize,
    frame_count: usize,
    time_seconds: f64,
) -> String {
    let phase = phoenix_phase(frame_index, frame_count);
    let mut controls = Vec::new();
    for track in tracks {
        let m = track.meters.get(frame_index).copied().unwrap_or_default();
        controls.push(format!(
            "\"{}\":{{\"lane\":\"{}\",\"activity_scale\":{:.6},\"meters\":{{\"rms\":{:.6},\"onset\":{:.6},\"bass\":{:.6},\"mid\":{:.6},\"high\":{:.6}}}}}",
            json_escape(track.id),
            json_escape(track.lane),
            track.activity_scale,
            m.rms,
            m.onset,
            m.bass,
            m.mid,
            m.high
        ));
    }
    format!(
        "{{\"schema_version\":\"truevision_phoenix_flats_frame_state_rs_v1\",\"renderer\":\"rust\",\"preset\":\"phoenix_from_the_flats_state_v0\",\"frame_index\":{},\"time_seconds\":{:.6},\"state_transform_arc\":{},\"state_layers\":[\"lineart_world_mask\",\"damaged_city_silhouette\",\"flats_river_depth_pulse\",\"memory_lattice\",\"dual_phoenix_vector_field\",\"impact_to_regrowth_wave\",\"forest_regrowth_mask\",\"clear_water_reflection_return\",\"stem_meter_trace\"],\"stem_controls\":{{{}}},\"render_lanes\":{{\"lead_vocals\":\"main_gold_ember_thread\",\"backing_vocals\":\"answering_rose_ember_thread\",\"drums\":\"pressure_hits\",\"bass\":\"flats_river_depth_pulse\",\"keyboard\":\"memory_lattice\",\"percussion\":\"ash_spark_nerves\",\"synth\":\"phoenix_heat_veil\",\"other\":\"steel_air_texture\"}},\"boundary\":{{\"stems_drive_visual_lanes\":true,\"external_visual_assets_used\":false,\"openai_generation_used\":false,\"generated_media_is_evidence\":false,\"state_transform_arc_logged\":true}}}}",
        frame_index,
        time_seconds,
        state_transform_arc_json(phase),
        controls.join(",")
    )
}

fn state_transform_arc_json(phase: PhoenixPhase) -> String {
    format!(
        "{{\"phase_index\":{},\"phase_name\":\"{}\",\"progress\":{:.6},\"lineart_ratio\":{:.6},\"destructive_fire_ratio\":{:.6},\"witness_expansion_ratio\":{:.6},\"dual_vector_pressure\":{:.6},\"impact_pressure\":{:.6},\"healing_color_ratio\":{:.6},\"regrowth_ratio\":{:.6},\"state_law\":\"audio_state_drives_visual_state_transform\"}}",
        phase.index,
        json_escape(phase.name),
        phase.progress,
        phase.lineart_ratio,
        phase.destructive_fire_ratio,
        phase.witness_expansion_ratio,
        phase.dual_vector_pressure,
        phase.impact_pressure,
        phase.healing_color_ratio,
        phase.regrowth_ratio
    )
}

fn manifest_json(
    args: &Args,
    tracks: &[StemTrack],
    video_path: &PathBuf,
    state_path: &PathBuf,
    meter_path: &PathBuf,
    frame_count: usize,
    wall_seconds: f64,
) -> String {
    let mut lanes = Vec::new();
    for track in tracks {
        lanes.push(format!(
            "{{\"id\":\"{}\",\"label\":\"{}\",\"lane\":\"{}\"}}",
            json_escape(track.id),
            json_escape(track.label),
            json_escape(track.lane)
        ));
    }
    let phase_names = PHASE_NAMES
        .iter()
        .map(|name| format!("\"{}\"", json_escape(name)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\n  \"schema_version\":\"truevision_phoenix_flats_manifest_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"phoenix_from_the_flats_state_v0\",\n  \"run_id\":\"{}\",\n  \"source\":{{\"master_audio\":\"{}\",\"stems_dir\":\"{}\"}},\n  \"stem_lanes\":[{}],\n  \"state_transform_arc\":{{\"phase_names\":[{}],\"phase_count\":6,\"timeline_basis\":\"normalized_frame_progress\",\"proof_law\":\"audio_state_drives_visual_state_transform\"}},\n  \"output\":{{\"mp4\":\"{}\",\"frame_state_jsonl\":\"{}\",\"stem_meter_summary_json\":\"{}\",\"width\":{},\"height\":{},\"fps\":{},\"duration_seconds\":{},\"frame_count\":{},\"encoder\":\"{}\",\"wall_seconds\":{:.6}}},\n  \"boundary\":{{\"python_render_loop\":false,\"external_visual_assets_used\":false,\"openai_generation_used\":false,\"art_imports_used\":false,\"stems_drive_visual_lanes\":true,\"literal_phoenix_spam\":false,\"generated_media_is_evidence\":false,\"state_transform_arc_logged\":true}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_dir.display().to_string()),
        lanes.join(","),
        phase_names,
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

fn receipt_json(
    args: &Args,
    video_path: &PathBuf,
    manifest_path: &PathBuf,
    state_path: &PathBuf,
    wall_seconds: f64,
) -> String {
    format!(
        "{{\n  \"schema_version\":\"truevision_phoenix_flats_receipt_rs_v1\",\n  \"renderer\":\"rust\",\n  \"preset\":\"phoenix_from_the_flats_state_v0\",\n  \"run_id\":\"{}\",\n  \"status\":\"complete\",\n  \"output_mp4\":\"{}\",\n  \"manifest_json\":\"{}\",\n  \"frame_state_jsonl\":\"{}\",\n  \"elapsed_seconds\":{:.6},\n  \"boundary\":{{\"stems_drive_visual_lanes\":true,\"external_visual_assets_used\":false,\"openai_generation_used\":false,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        wall_seconds
    )
}

fn slug(value: &str) -> String {
    let safe: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = safe.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "phoenix_from_the_flats_30s".to_string()
    } else {
        trimmed
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
