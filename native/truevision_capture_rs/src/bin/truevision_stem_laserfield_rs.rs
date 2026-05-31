use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use zip::ZipArchive;

const STEMS: [&str; 6] = ["Drums", "Bass", "Guitar", "Vocals", "Synth", "FX"];
const GUITAR_LASER_ALPHA: f32 = 0.35;
const VISIBLE_STEM_METER_OVERLAY: bool = true;

#[derive(Clone)]
struct Args {
    output_root: PathBuf,
    run_id: String,
    audio: PathBuf,
    stems_zip: PathBuf,
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
    name: &'static str,
    zip_entry: String,
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

    let (master_samples, sample_rate) = decode_wav_file(&args.audio, args.duration)?;
    let tracks = load_stem_tracks(&args.stems_zip, args.duration, sample_rate, args.fps, frame_count)?;
    let summary = build_meter_summary(&args, &tracks, sample_rate, frame_count);
    File::create(&meter_path)
        .map_err(|e| format!("meter summary open failed: {e}"))?
        .write_all(summary.as_bytes())
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
        let t = frame_index as f64 / args.fps as f64;
        render_frame(&args, &tracks, frame_index, t as f32, &mut frame);
        stdin
            .write_all(&frame)
            .map_err(|e| format!("ffmpeg write failed: {e}"))?;
        if frame_index % args.state_log_every == 0 {
            let state = frame_state_json(&tracks, frame_index, t);
            writeln!(state_file, "{state}").map_err(|e| format!("state write failed: {e}"))?;
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
    let manifest = manifest_json(
        &args,
        &video_path,
        &state_path,
        &meter_path,
        frame_count,
        sample_rate,
        wall_seconds,
    );
    File::create(&manifest_path)
        .map_err(|e| format!("manifest open failed: {e}"))?
        .write_all(manifest.as_bytes())
        .map_err(|e| format!("manifest write failed: {e}"))?;
    let receipt = receipt_json(&args, &video_path, &manifest_path, &state_path, wall_seconds);
    File::create(&receipt_path)
        .map_err(|e| format!("receipt open failed: {e}"))?
        .write_all(receipt.as_bytes())
        .map_err(|e| format!("receipt write failed: {e}"))?;
    println!(
        "{{\"renderer\":\"rust\",\"video_path\":\"{}\",\"manifest_path\":\"{}\",\"frame_count\":{},\"wall_seconds\":{:.3}}}",
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        frame_count,
        wall_seconds
    );
    let _ = master_samples;
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        output_root: PathBuf::from("outputs/stem_state_nightmare_rs"),
        run_id: "stem_laserfield_rs".to_string(),
        audio: PathBuf::new(),
        stems_zip: PathBuf::new(),
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
            "--stems-zip" => args.stems_zip = PathBuf::from(value),
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
    if args.stems_zip.as_os_str().is_empty() {
        return Err("--stems-zip is required".to_string());
    }
    if args.width < 64 || args.height < 64 {
        return Err("size too small".to_string());
    }
    if args.fps < 1 {
        return Err("fps must be positive".to_string());
    }
    if args.duration <= 0.0 {
        return Err("duration must be positive".to_string());
    }
    if args.state_log_every < 1 {
        return Err("state log interval must be positive".to_string());
    }
    Ok(args)
}

fn decode_wav_file(path: &PathBuf, duration: f64) -> Result<(Vec<f32>, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("wav read failed: {e}"))?;
    decode_wav_mono(&data, duration)
}

fn decode_wav_mono(data: &[u8], duration: f64) -> Result<(Vec<f32>, usize), String> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a PCM WAV file".to_string());
    }
    let mut cursor = 12_usize;
    let mut audio_format = 0_u16;
    let mut channels = 0_u16;
    let mut sample_rate = 0_u32;
    let mut bits_per_sample = 0_u16;
    let mut pcm: &[u8] = &[];
    while cursor + 8 <= data.len() {
        let id = &data[cursor..cursor + 4];
        let size = u32::from_le_bytes([
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]) as usize;
        cursor += 8;
        if cursor + size > data.len() {
            break;
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err("bad WAV fmt chunk".to_string());
                }
                audio_format = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
                channels = u16::from_le_bytes([data[cursor + 2], data[cursor + 3]]);
                sample_rate = u32::from_le_bytes([
                    data[cursor + 4],
                    data[cursor + 5],
                    data[cursor + 6],
                    data[cursor + 7],
                ]);
                bits_per_sample = u16::from_le_bytes([data[cursor + 14], data[cursor + 15]]);
            }
            b"data" => pcm = &data[cursor..cursor + size],
            _ => {}
        }
        cursor += size + (size % 2);
    }
    if audio_format != 1 {
        return Err(format!("unsupported WAV format {audio_format}; PCM required"));
    }
    if channels == 0 || sample_rate == 0 || pcm.is_empty() {
        return Err("WAV missing channels, sample rate, or data".to_string());
    }
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    if !matches!(bytes_per_sample, 1 | 2 | 3 | 4) {
        return Err(format!("unsupported WAV bits per sample: {bits_per_sample}"));
    }
    let frame_bytes = bytes_per_sample * channels as usize;
    let max_frames = (duration * sample_rate as f64).round().max(1.0) as usize;
    let available_frames = pcm.len() / frame_bytes;
    let frames = available_frames.min(max_frames);
    let mut out = Vec::with_capacity(frames);
    for frame_index in 0..frames {
        let mut sum = 0.0_f32;
        for channel in 0..channels as usize {
            let offset = frame_index * frame_bytes + channel * bytes_per_sample;
            let value = match bytes_per_sample {
                1 => (pcm[offset] as f32 - 128.0) / 128.0,
                2 => i16::from_le_bytes([pcm[offset], pcm[offset + 1]]) as f32 / 32768.0,
                3 => {
                    let raw = (pcm[offset] as i32)
                        | ((pcm[offset + 1] as i32) << 8)
                        | ((pcm[offset + 2] as i32) << 16);
                    let signed = if raw & 0x800000 != 0 { raw | !0xFFFFFF } else { raw };
                    signed as f32 / 8_388_608.0
                }
                4 => i32::from_le_bytes([
                    pcm[offset],
                    pcm[offset + 1],
                    pcm[offset + 2],
                    pcm[offset + 3],
                ]) as f32
                    / 2_147_483_648.0,
                _ => 0.0,
            };
            sum += value;
        }
        out.push((sum / channels as f32).clamp(-1.0, 1.0));
    }
    Ok((out, sample_rate as usize))
}

fn load_stem_tracks(
    stems_zip: &PathBuf,
    duration: f64,
    target_rate: usize,
    fps: usize,
    frame_count: usize,
) -> Result<Vec<StemTrack>, String> {
    let file = File::open(stems_zip).map_err(|e| format!("stems zip open failed: {e}"))?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("stems zip parse failed: {e}"))?;
    let mut tracks = Vec::new();
    for stem_name in STEMS {
        let mut selected_index = None;
        let needle = format!("({})", stem_name.to_ascii_lowercase());
        for index in 0..archive.len() {
            let entry = archive
                .by_index(index)
                .map_err(|e| format!("zip entry read failed: {e}"))?;
            let name = entry.name().to_ascii_lowercase();
            if name.ends_with(".wav") && name.contains(&needle) {
                selected_index = Some(index);
                break;
            }
        }
        let index = selected_index.ok_or_else(|| format!("missing stem WAV in zip: {stem_name}"))?;
        let mut entry = archive
            .by_index(index)
            .map_err(|e| format!("zip stem open failed: {e}"))?;
        let entry_name = entry.name().to_string();
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|e| format!("zip stem read failed: {e}"))?;
        drop(entry);
        let (samples, rate) = decode_wav_mono(&bytes, duration)?;
        let fitted = fit_rate_and_len(samples, rate, target_rate, (duration * target_rate as f64) as usize);
        let meters = compute_meters(&fitted, target_rate, fps, frame_count);
        tracks.push(StemTrack {
            name: stem_name,
            zip_entry: entry_name,
            meters,
        });
    }
    Ok(tracks)
}

fn fit_rate_and_len(samples: Vec<f32>, source_rate: usize, target_rate: usize, target_len: usize) -> Vec<f32> {
    let mut out = if source_rate == target_rate {
        samples
    } else {
        let duration = samples.len() as f64 / source_rate as f64;
        let new_len = (duration * target_rate as f64).round().max(1.0) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for index in 0..new_len {
            let source_pos = index as f64 * source_rate as f64 / target_rate as f64;
            let left = source_pos.floor() as usize;
            let right = (left + 1).min(samples.len().saturating_sub(1));
            let mix = (source_pos - left as f64) as f32;
            let value = samples.get(left).copied().unwrap_or(0.0) * (1.0 - mix)
                + samples.get(right).copied().unwrap_or(0.0) * mix;
            resampled.push(value);
        }
        resampled
    };
    out.resize(target_len, 0.0);
    out.truncate(target_len);
    out
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
            low = low * 0.985 + s * 0.015;
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

fn meters_for<'a>(tracks: &'a [StemTrack], name: &str, frame_index: usize) -> Meters {
    tracks
        .iter()
        .find(|track| track.name == name)
        .and_then(|track| track.meters.get(frame_index))
        .copied()
        .unwrap_or_default()
}

fn render_frame(args: &Args, tracks: &[StemTrack], frame_index: usize, t: f32, frame: &mut [u8]) {
    let drum_m = meters_for(tracks, "Drums", frame_index);
    let bass_m = meters_for(tracks, "Bass", frame_index);
    let guitar_m = meters_for(tracks, "Guitar", frame_index);
    let vocal_m = meters_for(tracks, "Vocals", frame_index);
    let synth_m = meters_for(tracks, "Synth", frame_index);
    let fx_m = meters_for(tracks, "FX", frame_index);

    let drums = drum_m.rms.max(drum_m.onset);
    let bass = bass_m.rms.max(bass_m.bass);
    let guitar = guitar_m.rms.max(guitar_m.mid);
    let guitar_mid = guitar_m.mid;
    let vocals = voice_pressure(vocal_m);
    let synth = synth_m.rms.max(synth_m.high);
    let fx = fx_m.rms.max(fx_m.onset);

    fill_background(frame, args.width, args.height, synth, guitar, fx);
    draw_depth_grid(frame, args.width, args.height, bass, drums, t);
    draw_center_state(frame, args.width, args.height, vocals, bass, synth, t);
    draw_laser_ribbons(frame, args.width, args.height, guitar, guitar_mid, synth, t);
    draw_shards(frame, args.width, args.height, drums, fx, t, frame_index);
    apply_mirror_prism(frame, args.width, args.height, synth, fx, t);
    draw_vocal_stem_lane(frame, args.width, args.height, vocal_m, t);
    let drum_onset = drum_m.onset;
    let fx_onset = fx_m.onset;
    if fx_onset > 0.62 || drum_onset > 0.72 {
        flash(frame, (fx_onset.max(drum_onset) * 26.0) as u8);
    }
    if VISIBLE_STEM_METER_OVERLAY {
        draw_visible_stem_meter_overlay(frame, args.width, args.height, tracks, frame_index);
    }
    draw_lower_banner(frame, args.width, args.height, build_generation_banner(), t);
}

fn voice_pressure(meters: Meters) -> f32 {
    meters
        .rms
        .max(meters.mid)
        .max(meters.high * 0.78)
        .max(meters.onset * 0.72)
        .clamp(0.0, 1.0)
}

fn fill_background(frame: &mut [u8], width: usize, height: usize, synth: f32, guitar: f32, fx: f32) {
    for y in 0..height {
        let yy = y as f32 / height.max(1) as f32;
        let r = ((0.020 + (1.0 - yy) * (0.052 + fx * 0.018)) * 255.0) as u8;
        let g = ((0.012 + (1.0 - yy) * (0.030 + guitar * 0.020)) * 255.0) as u8;
        let b = ((0.010 + (1.0 - yy) * (0.018 + synth * 0.018)) * 255.0) as u8;
        for x in 0..width {
            let i = (y * width + x) * 3;
            frame[i] = r;
            frame[i + 1] = g;
            frame[i + 2] = b;
        }
    }
}

fn draw_depth_grid(frame: &mut [u8], width: usize, height: usize, bass: f32, drums: f32, t: f32) {
    let horizon = (height as f32 * (0.54 + 0.035 * (t * 0.9).sin())) as i32;
    let vanish = (width as i32 / 2, horizon);
    let color = Color::new(
        (0.18 + bass * 0.30) * 255.0,
        (0.14 + bass * 0.20) * 255.0,
        (0.06 + bass * 0.12) * 255.0,
    );
    for line in 0..23 {
        let x = (-0.25 * width as f32 + (line as f32 / 22.0) * width as f32 * 1.5) as i32;
        draw_line(frame, width, height, vanish.0, vanish.1, x, height as i32, color, 0.24 + drums * 0.30, 1);
    }
    for row in 0..16 {
        let y_norm = row as f32 / 15.0;
        let y = horizon as f32 + y_norm.powf(1.7 + bass * 0.5) * (height as f32 - horizon as f32);
        draw_line(
            frame,
            width,
            height,
            0,
            y as i32,
            width as i32,
            y as i32,
            color,
            0.20 + drums * 0.35,
            1,
        );
    }
}

fn draw_center_state(frame: &mut [u8], width: usize, height: usize, vocals: f32, bass: f32, synth: f32, t: f32) {
    let cx = width as i32 / 2;
    let cy = (height as f32 * 0.52) as i32;
    let halo = Color::new(
        (0.50 + vocals * 0.42) * 255.0,
        (0.35 + synth * 0.35) * 255.0,
        (0.10 + vocals * 0.50) * 255.0,
    );
    fill_ellipse(
        frame,
        width,
        height,
        cx,
        cy,
        (70.0 + 190.0 * vocals + 90.0 * bass) as i32,
        (42.0 + 105.0 * vocals + 55.0 * synth) as i32,
        halo,
        0.08 + vocals * 0.11,
    );
    let core_h = (height as f32 * (0.19 + vocals * 0.14)) as i32;
    let core_w = (width as f32 * (0.018 + bass * 0.025)) as i32;
    darken_ellipse(frame, width, height, cx, cy, core_w, core_h, 0.56 + vocals * 0.25);
    darken_ellipse(frame, width, height, cx, cy - core_h / 2, core_w * 2, core_w * 2, 0.56 + vocals * 0.25);
    for ring in 0..3 {
        let radius = ((84 + ring * 62) as f32 * (1.0 + vocals * 0.24 + 0.08 * (t * 2.0 + ring as f32).sin())) as i32;
        draw_ellipse_outline(
            frame,
            width,
            height,
            cx,
            cy,
            radius,
            (radius as f32 * 0.42) as i32,
            t * 14.0 + ring as f32 * 38.0,
            Color::new(0.90 * 255.0, 0.75 * 255.0, 0.22 * 255.0),
            0.48,
        );
    }
}

fn draw_laser_ribbons(frame: &mut [u8], width: usize, height: usize, guitar: f32, guitar_mid: f32, synth: f32, t: f32) {
    let origins = [
        ((width as f32 * 0.08) as i32, (height as f32 * 0.18) as i32),
        ((width as f32 * 0.92) as i32, (height as f32 * 0.18) as i32),
        ((width as f32 * 0.18) as i32, (height as f32 * 0.48) as i32),
        ((width as f32 * 0.82) as i32, (height as f32 * 0.48) as i32),
        ((width as f32 * 0.50) as i32, (height as f32 * 0.12) as i32),
    ];
    let colors = [
        Color::new(0.92 * 255.0, 1.00 * 255.0, 0.15 * 255.0),
        Color::new(1.00 * 255.0, 0.48 * 255.0, 0.25 * 255.0),
        Color::new(1.00 * 255.0, 0.22 * 255.0, 0.92 * 255.0),
        Color::new(0.52 * 255.0, 0.22 * 255.0, 1.00 * 255.0),
        Color::new(0.16 * 255.0, 0.80 * 255.0, 1.00 * 255.0),
        Color::new(0.35 * 255.0, 1.00 * 255.0, 0.32 * 255.0),
    ];
    for index in 0..36 {
        let origin = origins[index % origins.len()];
        let angle = t * (0.6 + guitar_mid * 2.2) + index as f32 * 0.39 + (t * 1.7 + index as f32).sin() * 0.8;
        let radius = 0.42 + 0.28 * (t * 0.31 + index as f32 * 0.27).sin();
        let target = (
            (width as f32 * (0.50 + angle.cos() * radius)) as i32,
            (height as f32 * (0.52 + (angle * 0.73).sin() * (0.30 + synth * 0.15))) as i32,
        );
        let strength = (0.22 + guitar * 1.05 + 0.30 * (t * 4.0 + index as f32).sin()).max(0.0);
        let color = colors[index % colors.len()];
        draw_line(frame, width, height, origin.0, origin.1, target.0, target.1, color, strength * 0.13 * GUITAR_LASER_ALPHA, 8 + (guitar * 13.0) as i32);
        draw_line(frame, width, height, origin.0, origin.1, target.0, target.1, color, strength * 0.62 * GUITAR_LASER_ALPHA, 1 + (guitar * 3.0) as i32);
    }
}

fn draw_vocal_stem_lane(frame: &mut [u8], width: usize, height: usize, vocals: Meters, t: f32) {
    let pressure = voice_pressure(vocals);
    let cx = width as i32 / 2;
    let top = (height as f32 * 0.14) as i32;
    let bottom = (height as f32 * 0.86) as i32;
    let cy = (height as f32 * 0.52) as i32;
    let beam = Color::new(0.82 * 255.0, (0.92 + pressure * 0.08) * 255.0, 1.00 * 255.0);
    let amber = Color::new(1.00 * 255.0, 0.62 * 255.0, 0.12 * 255.0);

    draw_line(
        frame,
        width,
        height,
        cx,
        top,
        cx,
        bottom,
        beam,
        0.20 + pressure * 0.42,
        5 + (pressure * 18.0) as i32,
    );
    draw_line(
        frame,
        width,
        height,
        cx,
        top,
        cx,
        bottom,
        Color::new(1.0 * 255.0, 1.0 * 255.0, 1.0 * 255.0),
        0.18 + pressure * 0.52,
        1 + (pressure * 5.0) as i32,
    );

    let mut left_prev: Option<(i32, i32)> = None;
    let mut right_prev: Option<(i32, i32)> = None;
    for step in 0..96 {
        let n = step as f32 / 95.0;
        let y = top as f32 + n * (bottom - top) as f32;
        let envelope = (1.0 - ((n - 0.5).abs() * 1.55)).clamp(0.0, 1.0);
        let wave = (t * (5.2 + vocals.mid * 5.0) + step as f32 * (0.30 + vocals.high * 0.15)).sin();
        let amp = width as f32 * (0.035 + pressure * 0.075) * envelope;
        let spread = width as f32 * (0.075 + pressure * 0.045);
        let left = (cx as f32 - spread - wave * amp) as i32;
        let right = (cx as f32 + spread + wave * amp) as i32;
        let point_y = y as i32;
        if let Some((px, py)) = left_prev {
            draw_line(frame, width, height, px, py, left, point_y, beam, 0.30 + pressure * 0.46, 2);
        }
        if let Some((px, py)) = right_prev {
            draw_line(frame, width, height, px, py, right, point_y, beam, 0.30 + pressure * 0.46, 2);
        }
        if step % 8 == 0 {
            let rib = (width as f32 * (0.018 + pressure * 0.048) * envelope) as i32;
            draw_line(frame, width, height, cx - rib, point_y, cx + rib, point_y, amber, 0.18 + pressure * 0.28, 1);
        }
        left_prev = Some((left, point_y));
        right_prev = Some((right, point_y));
    }

    for ring in 0..4 {
        let scale = 1.0 + ring as f32 * 0.42;
        let rx = (width as f32 * (0.060 + pressure * 0.070) * scale) as i32;
        let ry = (height as f32 * (0.045 + pressure * 0.038) * scale) as i32;
        draw_ellipse_outline(
            frame,
            width,
            height,
            cx,
            cy,
            rx,
            ry,
            t * (34.0 + vocals.high * 42.0) + ring as f32 * 27.0,
            beam,
            0.24 + pressure * 0.24,
        );
    }
}

fn draw_visible_stem_meter_overlay(frame: &mut [u8], width: usize, height: usize, tracks: &[StemTrack], frame_index: usize) {
    let panel_x = 14;
    let panel_y = 14;
    let panel_w = (width as f32 * 0.25).clamp(190.0, 360.0) as i32;
    let row_h = 16;
    let panel_h = 24 + row_h * STEMS.len() as i32;
    fill_rect(
        frame,
        width,
        height,
        panel_x - 8,
        panel_y - 8,
        panel_w + 16,
        panel_h,
        Color::new(0.0, 0.0, 0.0),
        0.62,
    );
    draw_text(
        frame,
        width,
        height,
        "STEM LANES LIVE",
        panel_x,
        panel_y,
        1,
        Color::new(0.78 * 255.0, 0.94 * 255.0, 1.00 * 255.0),
        0.95,
    );
    for (index, stem) in STEMS.iter().enumerate() {
        let m = meters_for(tracks, stem, frame_index);
        let value = match *stem {
            "Drums" => m.rms.max(m.onset),
            "Bass" => m.rms.max(m.bass),
            "Guitar" => m.rms.max(m.mid),
            "Vocals" => voice_pressure(m),
            "Synth" => m.rms.max(m.high),
            "FX" => m.rms.max(m.onset),
            _ => m.rms,
        }
        .clamp(0.0, 1.0);
        let y = panel_y + 18 + index as i32 * row_h;
        draw_text(
            frame,
            width,
            height,
            stem,
            panel_x,
            y,
            1,
            stem_color(stem),
            0.90,
        );
        let bar_x = panel_x + 62;
        let bar_w = panel_w - 76;
        fill_rect(frame, width, height, bar_x, y + 2, bar_w, 8, Color::new(0.05 * 255.0, 0.06 * 255.0, 0.07 * 255.0), 0.80);
        fill_rect(frame, width, height, bar_x, y + 2, (bar_w as f32 * value) as i32, 8, stem_color(stem), 0.82);
    }
}

fn stem_color(stem: &str) -> Color {
    match stem {
        "Drums" => Color::new(1.00 * 255.0, 0.28 * 255.0, 0.20 * 255.0),
        "Bass" => Color::new(1.00 * 255.0, 0.68 * 255.0, 0.10 * 255.0),
        "Guitar" => Color::new(0.90 * 255.0, 1.00 * 255.0, 0.18 * 255.0),
        "Vocals" => Color::new(0.70 * 255.0, 0.92 * 255.0, 1.00 * 255.0),
        "Synth" => Color::new(0.40 * 255.0, 0.42 * 255.0, 1.00 * 255.0),
        "FX" => Color::new(1.00 * 255.0, 0.26 * 255.0, 0.92 * 255.0),
        _ => Color::new(0.70 * 255.0, 0.70 * 255.0, 0.70 * 255.0),
    }
}

fn draw_shards(frame: &mut [u8], width: usize, height: usize, drums: f32, fx: f32, t: f32, frame_index: usize) {
    let mut rng = XorShift::new(frame_index as u64 * 17 + 444);
    let shard_count = (10.0 + drums * 42.0 + fx * 34.0) as usize;
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.52;
    for index in 0..shard_count {
        let angle = (index as f32 / shard_count.max(1) as f32) * std::f32::consts::TAU
            + t * (0.5 + fx)
            + (rng.next_f32() - 0.5) * 0.16;
        let inner = 34.0 + drums * 80.0 + rng.next_f32() * 40.0;
        let outer = inner + 45.0 + rng.next_f32() * (140.0 + fx * 190.0);
        let p1 = (cx + angle.cos() * inner, cy + angle.sin() * inner);
        let p2 = (cx + (angle + 0.018).cos() * outer, cy + (angle + 0.018).sin() * outer);
        draw_line(
            frame,
            width,
            height,
            p1.0 as i32,
            p1.1 as i32,
            p2.0 as i32,
            p2.1 as i32,
            Color::new(255.0, (0.35 + drums * 0.55) * 255.0, (0.80 + fx * 0.20) * 255.0),
            0.38 + drums * 0.28,
            1 + (drums * 3.0) as i32,
        );
    }
}

fn apply_mirror_prism(frame: &mut [u8], width: usize, height: usize, synth: f32, fx: f32, t: f32) {
    if synth + fx < 0.08 {
        return;
    }
    let mix = 0.12 + synth * 0.22 + fx * 0.10;
    for y in 0..height {
        for x in width / 2..width {
            let mirror_x = width - 1 - x;
            let i = (y * width + x) * 3;
            let j = (y * width + mirror_x) * 3;
            for c in 0..3 {
                frame[i + c] = (frame[i + c] as f32 * (1.0 - mix) + frame[j + c] as f32 * mix) as u8;
            }
        }
    }
    let shift = ((synth * 12.0 + fx * 8.0) * (t * 1.9).sin()) as i32;
    if shift != 0 {
        channel_shift(frame, width, height, 0, shift);
        channel_shift(frame, width, height, 2, -shift);
    }
}

fn channel_shift(frame: &mut [u8], width: usize, height: usize, channel: usize, shift: i32) {
    let copy = frame.to_vec();
    for y in 0..height {
        for x in 0..width {
            let sx = ((x as i32 - shift).rem_euclid(width as i32)) as usize;
            frame[(y * width + x) * 3 + channel] = copy[(y * width + sx) * 3 + channel];
        }
    }
}

fn draw_lower_banner(frame: &mut [u8], width: usize, height: usize, text: &str, t: f32) {
    let banner_h = (height as f32 * 0.062).max(18.0) as usize;
    let top = height.saturating_sub(banner_h);
    for y in top..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            frame[i] = (frame[i] as f32 * 0.46) as u8;
            frame[i + 1] = (frame[i + 1] as f32 * 0.46) as u8;
            frame[i + 2] = (frame[i + 2] as f32 * 0.46) as u8;
        }
    }
    draw_line(
        frame,
        width,
        height,
        0,
        top as i32,
        width as i32,
        top as i32,
        Color::new(210.0, 190.0, 40.0),
        0.9,
        1,
    );
    let full = format!("{text}     {text}");
    let scale = (height / 260).max(1) as i32;
    let text_w = full.chars().count() as i32 * 6 * scale;
    let travel = text_w + width as i32;
    let mut x = width as i32 - ((t * 86.0) as i32).rem_euclid(travel.max(1));
    let y = top as i32 + ((banner_h as i32 - 7 * scale) / 2).max(2);
    while x < width as i32 {
        draw_text(frame, width, height, &full, x, y, scale, Color::new(248.0, 245.0, 215.0), 0.94);
        x += text_w + 80;
    }
}

fn draw_text(frame: &mut [u8], width: usize, height: usize, text: &str, x: i32, y: i32, scale: i32, color: Color, alpha: f32) {
    let mut cursor = x;
    for ch in text.chars() {
        draw_char(frame, width, height, ch, cursor, y, scale, color, alpha);
        cursor += 6 * scale;
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

fn draw_char(frame: &mut [u8], width: usize, height: usize, ch: char, x: i32, y: i32, scale: i32, color: Color, alpha: f32) {
    let glyph = glyph(ch);
    for (row, pattern) in glyph.iter().enumerate() {
        for (col, bit) in pattern.chars().enumerate() {
            if bit == '1' {
                for sy in 0..scale {
                    for sx in 0..scale {
                        blend_pixel(frame, width, height, x + col as i32 * scale + sx, y + row as i32 * scale + sy, color, alpha);
                    }
                }
            }
        }
    }
}

fn glyph(ch: char) -> [&'static str; 7] {
    match ch.to_ascii_uppercase() {
        'A' => ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        'B' => ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        'C' => ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        'D' => ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        'E' => ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        'F' => ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        'G' => ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
        'H' => ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        'I' => ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        'J' => ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
        'K' => ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        'L' => ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        'M' => ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        'N' => ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        'O' => ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        'P' => ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        'Q' => ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        'R' => ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        'S' => ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        'T' => ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        'U' => ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        'V' => ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        'W' => ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
        'X' => ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        'Y' => ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        'Z' => ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
        '/' => ["00001", "00001", "00010", "00100", "01000", "10000", "10000"],
        '-' => ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        ':' => ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
        _ => ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    }
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

fn darken_ellipse(frame: &mut [u8], width: usize, height: usize, cx: i32, cy: i32, rx: i32, ry: i32, strength: f32) {
    if rx <= 0 || ry <= 0 {
        return;
    }
    for y in (cy - ry).max(0)..=(cy + ry).min(height as i32 - 1) {
        for x in (cx - rx).max(0)..=(cx + rx).min(width as i32 - 1) {
            let dx = (x - cx) as f32 / rx as f32;
            let dy = (y - cy) as f32 / ry as f32;
            if dx * dx + dy * dy <= 1.0 {
                let i = (y as usize * width + x as usize) * 3;
                let keep = (1.0 - strength).clamp(0.0, 1.0);
                frame[i] = (frame[i] as f32 * keep) as u8;
                frame[i + 1] = (frame[i + 1] as f32 * keep) as u8;
                frame[i + 2] = (frame[i + 2] as f32 * keep) as u8;
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

fn flash(frame: &mut [u8], amount: u8) {
    for value in frame.iter_mut() {
        *value = value.saturating_add(amount);
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
        .arg("320k")
        .arg("-shortest")
        .arg(output_path)
        .status()
        .map_err(|e| format!("ffmpeg mux failed: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg mux failed with status {status}"))
    }
}

fn build_generation_banner() -> &'static str {
    "CORTEX EVOLVED  /  TRUEVISION STATE GENERATION  /  LOCAL FIRST  /  STEM-DRIVEN LIGHT  /  RECEIPT-BACKED CREATION  /  MUSIC STATE MADE VISIBLE"
}

fn build_meter_summary(args: &Args, tracks: &[StemTrack], sample_rate: usize, frame_count: usize) -> String {
    let mut stems = String::new();
    for (index, track) in tracks.iter().enumerate() {
        if index > 0 {
            stems.push(',');
        }
        stems.push_str(&format!(
            "{{\"stem_name\":\"{}\",\"zip_entry\":\"{}\",\"visual_lanes\":{}}}",
            track.name,
            json_escape(&track.zip_entry),
            visual_lanes_json(track.name)
        ));
    }
    format!(
        "{{\n  \"schema_version\":\"truevision_stem_meter_summary_rs_v1\",\n  \"renderer\":\"rust\",\n  \"master_audio\":\"{}\",\n  \"stems_zip\":\"{}\",\n  \"sample_rate\":{},\n  \"fps\":{},\n  \"duration_seconds\":{},\n  \"frame_count\":{},\n  \"stem_sources\":[{}],\n  \"boundary\":{{\"python_render_loop\":false,\"stems_drive_visual_lanes\":true,\"visible_stem_meter_overlay\":true,\"vocal_stem_visual_lane\":true,\"guitar_laser_alpha\":0.35}}\n}}\n",
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_zip.display().to_string()),
        sample_rate,
        args.fps,
        args.duration,
        frame_count,
        stems
    )
}

fn frame_state_json(tracks: &[StemTrack], frame_index: usize, time_seconds: f64) -> String {
    let mut controls = String::new();
    for (index, track) in tracks.iter().enumerate() {
        if index > 0 {
            controls.push(',');
        }
        let m = track.meters.get(frame_index).copied().unwrap_or_default();
        controls.push_str(&format!(
            "\"{}\":{{\"visual_lanes\":{},\"meters\":{{\"rms\":{:.6},\"onset\":{:.6},\"bass\":{:.6},\"mid\":{:.6},\"high\":{:.6}}}}}",
            track.name,
            visual_lanes_json(track.name),
            m.rms,
            m.onset,
            m.bass,
            m.mid,
            m.high
        ));
    }
    let vocal = meters_for(tracks, "Vocals", frame_index);
    let vocal_pressure = voice_pressure(vocal);
    format!(
        "{{\"schema_version\":\"truevision_stem_state_nightmare_frame_rs_v1\",\"renderer\":\"rust\",\"frame_index\":{},\"time_seconds\":{:.6},\"stem_controls\":{{{}}},\"render_lanes\":{{\"guitar_laser_alpha\":0.35,\"visible_stem_lanes\":[\"Drums\",\"Bass\",\"Guitar\",\"Vocals\",\"Synth\",\"FX\"],\"vocal_lane\":{{\"driver_stem\":\"Vocals\",\"visible\":true,\"shape\":\"center_voice_column_waveform\",\"voice_pressure\":{:.6}}}}},\"banner\":{{\"position\":\"lower_scrolling\",\"purpose\":\"identity_and_generation_tech\"}},\"boundary\":{{\"python_render_loop\":false,\"stems_drive_visual_lanes\":true,\"visible_stem_meter_overlay\":true,\"vocal_stem_visual_lane\":true,\"generated_media_is_evidence\":false}}}}",
        frame_index,
        time_seconds,
        controls,
        vocal_pressure
    )
}

fn visual_lanes_json(stem_name: &str) -> &'static str {
    match stem_name {
        "Drums" => "[\"impact_flash\",\"cut_shards\",\"glitch_gate\"]",
        "Bass" => "[\"depth_grid_pressure\",\"occlusion_core_breath\",\"floor_warp\"]",
        "Guitar" => "[\"laser_ribbons\",\"angular_state_transform\",\"edge_warp\"]",
        "Vocals" => "[\"central_shadow_axis\",\"halo_pressure\",\"focus_pull\"]",
        "Synth" => "[\"volumetric_color_field\",\"orbit_shells\",\"mirror_prism\"]",
        "FX" => "[\"spark_noise\",\"scanline_tears\",\"color_inversion_hits\"]",
        _ => "[]",
    }
}

fn manifest_json(
    args: &Args,
    video_path: &PathBuf,
    state_path: &PathBuf,
    meter_path: &PathBuf,
    frame_count: usize,
    sample_rate: usize,
    wall_seconds: f64,
) -> String {
    format!(
        "{{\n  \"schema_version\":\"truevision_stem_state_nightmare_manifest_rs_v1\",\n  \"renderer\":\"rust\",\n  \"run_id\":\"{}\",\n  \"source\":{{\"master_audio\":\"{}\",\"stems_zip\":\"{}\"}},\n  \"output\":{{\"mp4\":\"{}\",\"frame_state_jsonl\":\"{}\",\"stem_meter_summary_json\":\"{}\",\"width\":{},\"height\":{},\"fps\":{},\"duration_seconds\":{},\"frame_count\":{},\"encoder\":\"{}\",\"wall_seconds\":{:.6}}},\n  \"banner\":{{\"position\":\"lower_scrolling\",\"text\":\"{}\",\"purpose\":\"identity_and_generation_tech\"}},\n  \"sample_rate\":{},\n  \"boundary\":{{\"python_render_loop\":false,\"external_visual_assets_used\":false,\"openai_generation_used\":false,\"art_imports_used\":false,\"stems_drive_visual_lanes\":true,\"visible_stem_meter_overlay\":true,\"vocal_stem_visual_lane\":true,\"guitar_laser_alpha\":0.35,\"master_audio_drives_global_timing\":true,\"generated_media_is_evidence\":false}}\n}}\n",
        json_escape(&args.run_id),
        json_escape(&args.audio.display().to_string()),
        json_escape(&args.stems_zip.display().to_string()),
        json_escape(&video_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        json_escape(&meter_path.display().to_string()),
        args.width,
        args.height,
        args.fps,
        args.duration,
        frame_count,
        json_escape(&args.video_encoder),
        wall_seconds,
        json_escape(build_generation_banner()),
        sample_rate
    )
}

fn receipt_json(args: &Args, video_path: &PathBuf, manifest_path: &PathBuf, state_path: &PathBuf, wall_seconds: f64) -> String {
    format!(
        "{{\n  \"schema_version\":\"truevision_stem_state_nightmare_receipt_rs_v1\",\n  \"renderer\":\"rust\",\n  \"run_id\":\"{}\",\n  \"status\":\"complete\",\n  \"output_mp4\":\"{}\",\n  \"manifest_json\":\"{}\",\n  \"frame_state_jsonl\":\"{}\",\n  \"elapsed_seconds\":{:.6},\n  \"boundary\":{{\"python_render_loop\":false,\"stems_drive_visual_lanes\":true,\"visible_stem_meter_overlay\":true,\"vocal_stem_visual_lane\":true,\"guitar_laser_alpha\":0.35,\"generated_media_is_evidence\":false}}\n}}\n",
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
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "stem_laserfield_rs".to_string()
    } else {
        trimmed
    }
}
