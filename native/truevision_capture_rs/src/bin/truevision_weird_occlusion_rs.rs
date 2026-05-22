use std::env;
use std::ffi::c_void;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const WAVEFORM_BINS: usize = 96;

#[repr(C)]
#[allow(non_snake_case)]
struct ProcessMemoryCounters {
    cb: u32,
    PageFaultCount: u32,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
}

#[link(name = "psapi")]
unsafe extern "system" {
    fn GetProcessMemoryInfo(
        process: *mut c_void,
        counters: *mut ProcessMemoryCounters,
        size: u32,
    ) -> i32;
}

#[derive(Clone)]
struct Args {
    output_root: PathBuf,
    run_id: String,
    audio: Option<PathBuf>,
    sample_rate: usize,
    mux_audio: bool,
    scene_mode: String,
    palette: String,
    backdrop_image: Option<PathBuf>,
    width: usize,
    height: usize,
    fps: usize,
    duration: f64,
    crf: u8,
    video_encoder: String,
    bitrate: String,
    render_threads: usize,
    state_log_every: usize,
}

#[derive(Clone, Copy, Default)]
struct Color {
    b: f32,
    g: f32,
    r: f32,
}

impl Color {
    fn new(b: f32, g: f32, r: f32) -> Self {
        Self { b, g, r }
    }
}

#[derive(Clone, Copy)]
struct Rect {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Rect {
    fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

#[derive(Default)]
struct FrameStats {
    fog_coverage_sum: f64,
    fog_samples: usize,
    occluded_pixels_sum: u64,
    glow_pixels_sum: u64,
}

#[derive(Clone, Copy)]
struct AudioFeature {
    rms: f32,
    bass: f32,
    high: f32,
    beat: f32,
    waveform: [f32; WAVEFORM_BINS],
}

impl Default for AudioFeature {
    fn default() -> Self {
        Self {
            rms: 0.0,
            bass: 0.0,
            high: 0.0,
            beat: 0.0,
            waveform: [0.0; WAVEFORM_BINS],
        }
    }
}

struct Backdrop {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    content_rect: PanelRect,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let run_dir = args.output_root.join(&args.run_id);
    create_dir_all(&run_dir).map_err(|e| format!("create output dir failed: {e}"))?;
    let video_path = run_dir.join(format!("{}.mp4", args.run_id));
    let visual_path = if args.audio.is_some() && args.mux_audio {
        run_dir.join(format!("{}_visual_only.mp4", args.run_id))
    } else {
        video_path.clone()
    };
    let state_path = run_dir.join(format!("{}_frame_state.jsonl", args.run_id));
    let manifest_path = run_dir.join(format!("{}_manifest.json", args.run_id));

    let memory_start = process_memory_snapshot();
    let render_started = Instant::now();
    let frame_count = (args.duration * args.fps as f64).round() as usize;
    let audio_features = if let Some(audio_path) = &args.audio {
        let samples = decode_audio_mono(audio_path, args.sample_rate, args.duration)?;
        build_audio_features(&samples, args.sample_rate, args.fps, frame_count)
    } else {
        vec![AudioFeature::default(); frame_count]
    };
    let backdrop = if let Some(path) = &args.backdrop_image {
        Some(decode_backdrop_bgr(path, args.width, args.height)?)
    } else {
        None
    };
    let mut state_file =
        BufWriter::new(File::create(&state_path).map_err(|e| format!("state open failed: {e}"))?);

    let mut command = Command::new("ffmpeg");
    command
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("bgr24")
        .arg("-s")
        .arg(format!("{}x{}", args.width, args.height))
        .arg("-r")
        .arg(args.fps.to_string())
        .arg("-i")
        .arg("-")
        .arg("-an");
    configure_video_encoder(&mut command, &args);
    command.arg(&visual_path).stdin(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|e| format!("ffmpeg start failed: {e}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "ffmpeg stdin missing".to_string())?;
    let mut frame = vec![0_u8; args.width * args.height * 3];
    let mut stats = FrameStats::default();

    for frame_index in 0..frame_count {
        let time_seconds = frame_index as f64 / args.fps as f64;
        let audio = audio_features.get(frame_index).copied().unwrap_or_default();
        let frame_meta = render_frame(
            &args,
            backdrop.as_ref(),
            time_seconds,
            frame_index,
            audio,
            &mut frame,
            &mut stats,
        );
        stdin
            .write_all(&frame)
            .map_err(|e| format!("ffmpeg write failed: {e}"))?;
        if frame_index % args.state_log_every == 0 {
            writeln!(state_file, "{frame_meta}").map_err(|e| format!("state write failed: {e}"))?;
        }
    }
    drop(stdin);
    let code = child
        .wait()
        .map_err(|e| format!("ffmpeg wait failed: {e}"))?;
    if !code.success() {
        return Err(format!("ffmpeg failed with status {code}"));
    }
    if let Some(audio_path) = &args.audio {
        if args.mux_audio {
            mux_audio(&visual_path, audio_path, &video_path, args.duration)?;
        }
    }
    state_file
        .flush()
        .map_err(|e| format!("state flush failed: {e}"))?;

    let wall_seconds = render_started.elapsed().as_secs_f64();
    let memory_end = process_memory_snapshot();
    write_manifest(
        &args,
        &video_path,
        &visual_path,
        &state_path,
        &manifest_path,
        frame_count,
        wall_seconds,
        memory_start,
        memory_end,
        &stats,
    )?;
    println!(
        "{{\"video_path\":\"{}\",\"manifest_path\":\"{}\",\"frame_count\":{},\"duration_seconds\":{:.3},\"wall_seconds\":{:.3}}}",
        json_escape(&video_path.display().to_string()),
        json_escape(&manifest_path.display().to_string()),
        frame_count,
        args.duration,
        wall_seconds
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        output_root: PathBuf::from("outputs/weird_occlusion_rs"),
        run_id: "mist_engine_occlusion_demo_rs".to_string(),
        audio: None,
        sample_rate: 44_100,
        mux_audio: true,
        scene_mode: "mist_engine".to_string(),
        palette: "ember_city".to_string(),
        backdrop_image: None,
        width: 1280,
        height: 720,
        fps: 30,
        duration: 24.0,
        crf: 16,
        video_encoder: "libx264".to_string(),
        bitrate: "24M".to_string(),
        render_threads: thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
        state_log_every: 30,
    };
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--output-root" => args.output_root = PathBuf::from(&value),
            "--run-id" => args.run_id = slug(&value),
            "--audio" => args.audio = Some(PathBuf::from(value)),
            "--sample-rate" => {
                args.sample_rate = value.parse().map_err(|_| "bad sample rate".to_string())?
            }
            "--mux-audio" => args.mux_audio = parse_bool(&value)?,
            "--scene-mode" => args.scene_mode = slug(&value),
            "--palette" => args.palette = slug(&value),
            "--backdrop-image" => args.backdrop_image = Some(PathBuf::from(value)),
            "--size" => {
                let size = parse_size(&value)?;
                args.width = size.0;
                args.height = size.1;
            }
            "--width" => args.width = value.parse().map_err(|_| "bad width".to_string())?,
            "--height" => args.height = value.parse().map_err(|_| "bad height".to_string())?,
            "--fps" => args.fps = value.parse().map_err(|_| "bad fps".to_string())?,
            "--duration" => {
                args.duration = value.parse().map_err(|_| "bad duration".to_string())?
            }
            "--crf" => args.crf = value.parse().map_err(|_| "bad crf".to_string())?,
            "--video-encoder" => args.video_encoder = slug(&value),
            "--bitrate" => args.bitrate = value,
            "--render-threads" => {
                args.render_threads = value
                    .parse()
                    .map_err(|_| "bad render threads".to_string())?
            }
            "--state-log-every" => {
                args.state_log_every = value
                    .parse()
                    .map_err(|_| "bad state log interval".to_string())?
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if args.width < 320 || args.height < 180 {
        return Err("size too small".to_string());
    }
    if args.fps < 1 {
        return Err("fps must be positive".to_string());
    }
    if args.render_threads < 1 {
        return Err("render threads must be positive".to_string());
    }
    if args.state_log_every < 1 {
        return Err("state log interval must be positive".to_string());
    }
    Ok(args)
}

fn parse_size(value: &str) -> Result<(usize, usize), String> {
    let cleaned = value.replace('x', ",");
    let parts: Vec<&str> = cleaned.split(',').collect();
    if parts.len() != 2 {
        return Err("size must look like 1280x720".to_string());
    }
    let width = parts[0].parse().map_err(|_| "bad size width".to_string())?;
    let height = parts[1]
        .parse()
        .map_err(|_| "bad size height".to_string())?;
    Ok((width, height))
}

fn configure_video_encoder(command: &mut Command, args: &Args) {
    command.arg("-c:v").arg(&args.video_encoder);
    if args.video_encoder == "libx264" || args.video_encoder == "libx264rgb" {
        command
            .arg("-preset")
            .arg("veryfast")
            .arg("-crf")
            .arg(args.crf.to_string())
            .arg("-pix_fmt")
            .arg("yuv420p");
    } else {
        command
            .arg("-vf")
            .arg("format=nv12")
            .arg("-b:v")
            .arg(&args.bitrate)
            .arg("-maxrate")
            .arg(&args.bitrate)
            .arg("-pix_fmt")
            .arg("nv12");
    }
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("bad bool: {value}")),
    }
}

fn decode_audio_mono(
    audio_path: &PathBuf,
    sample_rate: usize,
    duration: f64,
) -> Result<Vec<f32>, String> {
    let output = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(audio_path)
        .arg("-t")
        .arg(format!("{duration:.6}"))
        .arg("-f")
        .arg("s16le")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(sample_rate.to_string())
        .arg("-")
        .output()
        .map_err(|e| format!("ffmpeg audio decode failed: {e}"))?;
    if !output.status.success() {
        return Err("ffmpeg audio decode returned non-zero status".to_string());
    }
    let mut cursor = std::io::Cursor::new(output.stdout);
    let mut bytes = Vec::new();
    cursor
        .read_to_end(&mut bytes)
        .map_err(|e| format!("audio buffer read failed: {e}"))?;
    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let value = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(value as f32 / 32768.0);
    }
    Ok(samples)
}

fn decode_backdrop_bgr(path: &PathBuf, width: usize, height: usize) -> Result<Backdrop, String> {
    let source_size = probe_image_size(path).unwrap_or((width, height));
    let content_rect = fitted_content_rect(source_size.0, source_size.1, width, height);
    let filter = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    );
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
        .arg("bgr24")
        .arg("-")
        .output()
        .map_err(|e| format!("ffmpeg backdrop decode failed: {e}"))?;
    if !output.status.success() {
        return Err("ffmpeg backdrop decode returned non-zero status".to_string());
    }
    let expected = width * height * 3;
    if output.stdout.len() != expected {
        return Err(format!(
            "backdrop decode size mismatch: got {}, expected {}",
            output.stdout.len(),
            expected
        ));
    }
    Ok(Backdrop {
        pixels: output.stdout,
        width,
        height,
        content_rect,
    })
}

fn probe_image_size(path: &PathBuf) -> Result<(usize, usize), String> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height")
        .arg("-of")
        .arg("csv=p=0:s=x")
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe image size failed: {e}"))?;
    if !output.status.success() {
        return Err("ffprobe image size returned non-zero status".to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let dims = text.trim();
    let Some((w, h)) = dims.split_once('x') else {
        return Err(format!("bad image size output: {dims}"));
    };
    let width = w.parse().map_err(|_| format!("bad image width: {w}"))?;
    let height = h.parse().map_err(|_| format!("bad image height: {h}"))?;
    Ok((width, height))
}

fn fitted_content_rect(
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
) -> PanelRect {
    let source_aspect = source_width as f32 / source_height.max(1) as f32;
    let target_aspect = width as f32 / height.max(1) as f32;
    if source_aspect > target_aspect {
        let scaled_h = width as f32 / source_aspect;
        let y0 = (height as f32 - scaled_h) * 0.5 / height as f32;
        PanelRect {
            x0: 0.0,
            y0,
            x1: 1.0,
            y1: y0 + scaled_h / height as f32,
        }
    } else {
        let scaled_w = height as f32 * source_aspect;
        let x0 = (width as f32 - scaled_w) * 0.5 / width as f32;
        PanelRect {
            x0,
            y0: 0.0,
            x1: x0 + scaled_w / width as f32,
            y1: 1.0,
        }
    }
}

fn build_audio_features(
    samples: &[f32],
    sample_rate: usize,
    fps: usize,
    frame_count: usize,
) -> Vec<AudioFeature> {
    if samples.is_empty() {
        return vec![AudioFeature::default(); frame_count];
    }
    let window = ((sample_rate as f32 * 0.080).round() as usize).max(256);
    let half = window / 2;
    let mut raw_rms = vec![0.0_f32; frame_count];
    let mut raw_bass = vec![0.0_f32; frame_count];
    let mut raw_high = vec![0.0_f32; frame_count];
    let mut raw_waveform = vec![[0.0_f32; WAVEFORM_BINS]; frame_count];
    for frame_index in 0..frame_count {
        let center = ((frame_index as f64 / fps as f64) * sample_rate as f64).round() as isize;
        let mut energy = 0.0_f64;
        let mut low_accum = 0.0_f64;
        let mut high_accum = 0.0_f64;
        let mut previous = 0.0_f32;
        let mut count = 0_usize;
        for offset in 0..window {
            let sample_index = center + offset as isize - half as isize;
            let sample = if sample_index >= 0 && (sample_index as usize) < samples.len() {
                samples[sample_index as usize]
            } else {
                0.0
            };
            energy += (sample as f64) * (sample as f64);
            low_accum += sample.abs() as f64;
            high_accum += (sample - previous).abs() as f64;
            previous = sample;
            count += 1;
        }
        let denom = count.max(1) as f64;
        raw_rms[frame_index] = (energy / denom).sqrt() as f32;
        raw_bass[frame_index] = (low_accum / denom) as f32;
        raw_high[frame_index] = (high_accum / denom) as f32;
        raw_waveform[frame_index] = frame_waveform_bins(samples, center, window);
    }
    normalize_audio_features(&raw_rms, &raw_bass, &raw_high, &raw_waveform)
}

fn frame_waveform_bins(samples: &[f32], center: isize, window: usize) -> [f32; WAVEFORM_BINS] {
    let mut bins = [0.0_f32; WAVEFORM_BINS];
    let local_radius = 2_isize;
    for (bin_index, slot) in bins.iter_mut().enumerate() {
        let t = if WAVEFORM_BINS <= 1 {
            0.5
        } else {
            bin_index as f32 / (WAVEFORM_BINS - 1) as f32
        };
        let sample_center = center + ((t - 0.5) * window as f32).round() as isize;
        let mut accum = 0.0_f32;
        let mut weight_sum = 0.0_f32;
        for offset in -local_radius..=local_radius {
            let sample_index = sample_center + offset;
            let sample = if sample_index >= 0 && (sample_index as usize) < samples.len() {
                samples[sample_index as usize]
            } else {
                0.0
            };
            let distance = (offset as f32).abs() / local_radius.max(1) as f32;
            let weight = (1.0 - distance * 0.45).clamp(0.0, 1.0);
            accum += sample * weight;
            weight_sum += weight;
        }
        *slot = accum / weight_sum.max(0.0001);
    }
    bins
}

fn normalize_audio_features(
    raw_rms: &[f32],
    raw_bass: &[f32],
    raw_high: &[f32],
    raw_waveform: &[[f32; WAVEFORM_BINS]],
) -> Vec<AudioFeature> {
    let rms_scale = percentile(raw_rms, 0.95).max(0.000_001);
    let bass_scale = percentile(raw_bass, 0.95).max(0.000_001);
    let high_scale = percentile(raw_high, 0.95).max(0.000_001);
    let mut features = Vec::with_capacity(raw_rms.len());
    let mut smooth = 0.0_f32;
    for index in 0..raw_rms.len() {
        let rms = (raw_rms[index] / rms_scale).clamp(0.0, 1.0);
        let bass = (raw_bass[index] / bass_scale).clamp(0.0, 1.0);
        let high = (raw_high[index] / high_scale).clamp(0.0, 1.0);
        let previous = smooth;
        smooth = smooth * 0.72 + rms * 0.28;
        let beat = (rms - previous)
            .max(0.0)
            .mul_add(2.8, high * 0.18)
            .clamp(0.0, 1.0);
        let mut waveform = raw_waveform
            .get(index)
            .copied()
            .unwrap_or([0.0; WAVEFORM_BINS]);
        let max_abs = waveform
            .iter()
            .fold(0.0_f32, |acc, sample| acc.max(sample.abs()))
            .max(0.0001);
        let visible_gain = (0.42 + 0.58 * rms + 0.22 * high).clamp(0.35, 1.18);
        for sample in &mut waveform {
            *sample = ((*sample / max_abs) * visible_gain).clamp(-1.0, 1.0);
        }
        features.push(AudioFeature {
            rms,
            bass,
            high,
            beat,
            waveform,
        });
    }
    features
}

fn percentile(values: &[f32], pct: f32) -> f32 {
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return 1.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((sorted.len() - 1) as f32 * pct.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn mux_audio(
    visual_path: &PathBuf,
    audio_path: &PathBuf,
    final_path: &PathBuf,
    duration: f64,
) -> Result<(), String> {
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(visual_path)
        .arg("-i")
        .arg(audio_path)
        .arg("-t")
        .arg(format!("{duration:.6}"))
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
        .arg(final_path)
        .status()
        .map_err(|e| format!("ffmpeg mux failed: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg mux failed with status {status}"))
    }
}

fn render_frame(
    args: &Args,
    backdrop: Option<&Backdrop>,
    time_seconds: f64,
    frame_index: usize,
    audio: AudioFeature,
    frame: &mut [u8],
    stats: &mut FrameStats,
) -> String {
    if args.scene_mode == "spectrum_backdrop" {
        return render_spectrum_backdrop_frame(
            args,
            backdrop,
            time_seconds,
            frame_index,
            audio,
            frame,
            stats,
        );
    }
    if args.scene_mode == "lyric_city" {
        return render_lyric_city_frame(args, time_seconds, frame_index, audio, frame, stats);
    }
    if args.scene_mode == "abstract_symphony" || args.scene_mode == "symphony" {
        return render_abstract_symphony_frame(
            args,
            time_seconds,
            frame_index,
            audio,
            frame,
            stats,
        );
    }
    if args.scene_mode == "warp_laser_field" || args.scene_mode == "laser_warp" {
        return render_warp_laser_field_frame(args, time_seconds, frame_index, audio, frame, stats);
    }
    if args.scene_mode == "memory_cathedral" || args.scene_mode == "fade_away_memory_cathedral" {
        return render_memory_cathedral_frame(args, time_seconds, frame_index, audio, frame, stats);
    }
    let width = args.width;
    let height = args.height;
    let horizon = (height as f32 * (0.45 + 0.025 * (time_seconds * 0.23).sin() as f32)) as i32;
    let pulse = (0.24 + 0.36 * (0.5 + 0.5 * (time_seconds * 1.9).sin() as f32) + 0.40 * audio.beat)
        .clamp(0.0, 1.0);
    let veil =
        (0.35 + 0.25 * (0.5 + 0.5 * (time_seconds * 0.31 + 1.7).sin() as f32) + 0.40 * audio.rms)
            .clamp(0.0, 1.0);

    let pillars = build_pillars(width, height, time_seconds);
    let silhouettes = build_silhouettes(width, height, time_seconds);
    let mut occluded_pixels = 0_u64;
    let mut glow_pixels = 0_u64;
    let mut fog_accum = 0.0_f64;

    for y in 0..height {
        let yf = y as f32 / height as f32;
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let mut color = background_color(xf, yf, horizon as f32 / height as f32, time_seconds);
            let depth = depth_at(xf, yf, time_seconds);
            apply_ground_reflection(&mut color, xf, yf, time_seconds, pulse);
            let portal =
                portal_glow(xf, yf, time_seconds) * (0.70 + 0.55 * audio.bass + 0.35 * audio.beat);
            if portal > 0.0 {
                glow_pixels += 1;
                blend_add(
                    &mut color,
                    Color::new(38.0, 96.0, 190.0),
                    portal * (0.35 + 0.45 * pulse),
                );
            }
            let silhouette_alpha = silhouette_alpha(x as i32, y as i32, &silhouettes);
            if silhouette_alpha > 0.0 {
                blend(&mut color, Color::new(6.0, 8.0, 11.0), silhouette_alpha);
            }
            let pillar_alpha = pillar_alpha(x as i32, y as i32, &pillars);
            if pillar_alpha > 0.0 {
                occluded_pixels += 1;
                blend(&mut color, Color::new(3.0, 4.0, 7.0), pillar_alpha);
                let rim = pillar_rim(x as i32, y as i32, &pillars);
                if rim > 0.0 {
                    blend_add(
                        &mut color,
                        Color::new(16.0, 42.0, 86.0),
                        rim * (0.4 + 0.3 * pulse),
                    );
                }
            }
            let fog =
                fog_density(xf, yf, depth, time_seconds, frame_index) * (0.72 + 0.52 * audio.rms);
            fog_accum += fog as f64;
            apply_fog(&mut color, fog, veil, yf);
            let ember = ember_field(xf, yf, time_seconds, audio.high, audio.beat);
            if ember > 0.0 {
                blend_add(&mut color, Color::new(5.0, 78.0, 210.0), ember);
            }
            write_pixel(frame, width, x, y, color);
        }
    }

    stats.fog_coverage_sum += fog_accum / (width * height) as f64;
    stats.fog_samples += 1;
    stats.occluded_pixels_sum += occluded_pixels;
    stats.glow_pixels_sum += glow_pixels;

    format!(
        "{{\"frame_index\":{},\"time_seconds\":{:.6},\"scene\":\"mist_engine_occlusion_corridor\",\"occlusion\":\"foreground_pillars_and_silhouettes\",\"audio\":{{\"rms\":{:.6},\"bass\":{:.6},\"high\":{:.6},\"beat\":{:.6}}},\"fog_mean\":{:.6},\"occluded_pixels\":{},\"glow_pixels\":{},\"state_layers\":[\"depth_bands\",\"portal_glow_audio_pressure\",\"pillar_occlusion\",\"moving_silhouette_occlusion\",\"mist_veils_audio_pressure\",\"ember_pressure_highbeat\",\"wet_reflection_bass\"]}}",
        frame_index,
        time_seconds,
        audio.rms,
        audio.bass,
        audio.high,
        audio.beat,
        fog_accum / (width * height) as f64,
        occluded_pixels,
        glow_pixels
    )
}

fn render_spectrum_backdrop_frame(
    args: &Args,
    backdrop: Option<&Backdrop>,
    time_seconds: f64,
    frame_index: usize,
    audio: AudioFeature,
    frame: &mut [u8],
    stats: &mut FrameStats,
) -> String {
    let width = args.width;
    let height = args.height;
    let panel = output_panel_rect(backdrop, analyzer_panel_rect());
    let vocal_panel = output_panel_rect(backdrop, vocal_wave_panel_rect());
    let radar_panel = output_panel_rect(backdrop, soundfield_panel_rect());
    let mark_panel = output_panel_rect(backdrop, prototype_mark_panel_rect());
    let bars = 36_usize;
    let video_duration_label = format!("DUR {:.0}S", args.duration);
    let mut glow_pixels = 0_u64;
    let mut occluded_pixels = 0_u64;
    let fog_accum = 0.0_f64;
    let palette = args.palette.as_str();

    for y in 0..height {
        let yf = y as f32 / height as f32;
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let mut color = if let Some(plate) = backdrop {
                backdrop_color(plate, x, y)
            } else {
                lyric_city_background(xf, yf, time_seconds, 0.55, 0.55, palette)
            };

            if let Some(plate) = backdrop {
                if let Some((u, v)) = backdrop_source_uv(plate, xf, yf) {
                    let electric = existing_electric_intensity(u, v, color, time_seconds, audio);
                    if electric.alpha > 0.0 {
                        blend_add(&mut color, electric.color, electric.alpha);
                        glow_pixels += 1;
                    }
                }
            }

            let analyzer_box = analyzer_black_box(xf, yf, panel);
            if analyzer_box.alpha > 0.0 {
                blend(&mut color, analyzer_box.color, analyzer_box.alpha);
            }

            let analyzer = analyzer_panel_light(
                xf,
                yf,
                panel,
                bars,
                time_seconds,
                audio.rms,
                audio.bass,
                audio.high,
                audio.beat,
            );
            if analyzer.alpha > 0.0 {
                blend_add(&mut color, analyzer.color, analyzer.alpha);
                glow_pixels += 1;
            }

            let vocal_box = waveform_black_box(xf, yf, vocal_panel);
            if vocal_box.alpha > 0.0 {
                blend(&mut color, vocal_box.color, vocal_box.alpha);
            }

            let vocal_wave = vocal_wave_panel_light(xf, yf, vocal_panel, audio);
            if vocal_wave.alpha > 0.0 {
                blend_add(&mut color, vocal_wave.color, vocal_wave.alpha);
                glow_pixels += 1;
            }

            let soundfield = headphone_soundfield_light(
                xf,
                yf,
                radar_panel,
                time_seconds,
                audio.rms,
                audio.bass,
                audio.high,
                audio.beat,
            );
            if soundfield.alpha > 0.0 {
                blend_add(&mut color, soundfield.color, soundfield.alpha);
                glow_pixels += 1;
            }

            let mark = prototype_mark_light(xf, yf, mark_panel, time_seconds, audio.beat);
            if mark.alpha > 0.0 {
                blend_add(&mut color, mark.color, mark.alpha);
                glow_pixels += 1;
            }

            if let Some(plate) = backdrop {
                let status = letterbox_status_light(
                    xf,
                    yf,
                    plate.content_rect,
                    time_seconds,
                    audio.beat,
                    &video_duration_label,
                );
                if status.alpha > 0.0 {
                    blend_add(&mut color, status.color, status.alpha);
                    glow_pixels += 1;
                }
            }

            if yf > 0.90 {
                occluded_pixels += 1;
            }
            write_pixel(frame, width, x, y, color);
        }
    }

    stats.fog_coverage_sum += fog_accum / (width * height) as f64;
    stats.fog_samples += 1;
    stats.occluded_pixels_sum += occluded_pixels;
    stats.glow_pixels_sum += glow_pixels;

    format!(
        "{{\"frame_index\":{},\"time_seconds\":{:.6},\"scene\":\"spectrum_backdrop\",\"palette\":\"{}\",\"audio\":{{\"rms\":{:.6},\"bass\":{:.6},\"high\":{:.6},\"beat\":{:.6},\"vocal_presence\":{:.6}}},\"backdrop_loaded\":{},\"backdrop_content_rect\":{{\"x0\":{:.6},\"y0\":{:.6},\"x1\":{:.6},\"y1\":{:.6}}},\"analyzer_panel\":{{\"x0\":{:.6},\"y0\":{:.6},\"x1\":{:.6},\"y1\":{:.6}}},\"vocal_wave_panel\":{{\"x0\":{:.6},\"y0\":{:.6},\"x1\":{:.6},\"y1\":{:.6}}},\"soundfield_panel\":{{\"x0\":{:.6},\"y0\":{:.6},\"x1\":{:.6},\"y1\":{:.6}}},\"prototype_mark_panel\":{{\"x0\":{:.6},\"y0\":{:.6},\"x1\":{:.6},\"y1\":{:.6}}},\"status_mark\":\"TRUEVISION_COPYRIGHT_2026_PROTOTYPE_CPU_RAM_MANIFEST\",\"fog_mean\":{:.6},\"occluded_pixels\":{},\"glow_pixels\":{},\"state_layers\":[\"full_backdrop_letterbox_fit\",\"existing_electric_intensity_only\",\"black_replacement_analyzer_box\",\"blue_gold_schema_analyzer\",\"black_replacement_waveform_box\",\"real_audio_waveform_panel\",\"approx_headphone_soundfield_radar\",\"truevision_prototype_mark\",\"letterbox_status_mark\",\"photo_schema_purple\"]}}",
        frame_index,
        time_seconds,
        palette,
        audio.rms,
        audio.bass,
        audio.high,
        audio.beat,
        vocal_presence(audio),
        backdrop.is_some(),
        backdrop.map(|b| b.content_rect.x0).unwrap_or(0.0),
        backdrop.map(|b| b.content_rect.y0).unwrap_or(0.0),
        backdrop.map(|b| b.content_rect.x1).unwrap_or(1.0),
        backdrop.map(|b| b.content_rect.y1).unwrap_or(1.0),
        panel.x0,
        panel.y0,
        panel.x1,
        panel.y1,
        vocal_panel.x0,
        vocal_panel.y0,
        vocal_panel.x1,
        vocal_panel.y1,
        radar_panel.x0,
        radar_panel.y0,
        radar_panel.x1,
        radar_panel.y1,
        mark_panel.x0,
        mark_panel.y0,
        mark_panel.x1,
        mark_panel.y1,
        fog_accum / (width * height) as f64,
        occluded_pixels,
        glow_pixels
    )
}

#[derive(Clone, Copy)]
struct AnalyzerLight {
    alpha: f32,
    color: Color,
}

#[derive(Clone, Copy)]
struct PanelRect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

fn analyzer_panel_rect() -> PanelRect {
    PanelRect {
        x0: 0.016,
        y0: 0.255,
        x1: 0.162,
        y1: 0.392,
    }
}

fn vocal_wave_panel_rect() -> PanelRect {
    PanelRect {
        x0: 0.831,
        y0: 0.491,
        x1: 0.987,
        y1: 0.573,
    }
}

fn soundfield_panel_rect() -> PanelRect {
    PanelRect {
        x0: 0.828,
        y0: 0.584,
        x1: 0.978,
        y1: 0.742,
    }
}

fn prototype_mark_panel_rect() -> PanelRect {
    PanelRect {
        x0: 0.806,
        y0: 0.756,
        x1: 0.978,
        y1: 0.828,
    }
}

fn output_panel_rect(backdrop: Option<&Backdrop>, source_panel: PanelRect) -> PanelRect {
    if let Some(plate) = backdrop {
        PanelRect {
            x0: plate.content_rect.x0
                + source_panel.x0 * (plate.content_rect.x1 - plate.content_rect.x0),
            y0: plate.content_rect.y0
                + source_panel.y0 * (plate.content_rect.y1 - plate.content_rect.y0),
            x1: plate.content_rect.x0
                + source_panel.x1 * (plate.content_rect.x1 - plate.content_rect.x0),
            y1: plate.content_rect.y0
                + source_panel.y1 * (plate.content_rect.y1 - plate.content_rect.y0),
        }
    } else {
        source_panel
    }
}

fn backdrop_source_uv(backdrop: &Backdrop, x: f32, y: f32) -> Option<(f32, f32)> {
    if x < backdrop.content_rect.x0
        || x > backdrop.content_rect.x1
        || y < backdrop.content_rect.y0
        || y > backdrop.content_rect.y1
    {
        None
    } else {
        Some((
            ((x - backdrop.content_rect.x0)
                / (backdrop.content_rect.x1 - backdrop.content_rect.x0).max(0.0001))
            .clamp(0.0, 1.0),
            ((y - backdrop.content_rect.y0)
                / (backdrop.content_rect.y1 - backdrop.content_rect.y0).max(0.0001))
            .clamp(0.0, 1.0),
        ))
    }
}

fn backdrop_color(backdrop: &Backdrop, x: usize, y: usize) -> Color {
    let sx = x.min(backdrop.width.saturating_sub(1));
    let sy = y.min(backdrop.height.saturating_sub(1));
    let idx = (sy * backdrop.width + sx) * 3;
    Color::new(
        backdrop.pixels[idx] as f32,
        backdrop.pixels[idx + 1] as f32,
        backdrop.pixels[idx + 2] as f32,
    )
}

fn analyzer_black_box(x: f32, y: f32, panel: PanelRect) -> AnalyzerLight {
    let pad_x = (panel.x1 - panel.x0) * 0.030;
    let pad_y = (panel.y1 - panel.y0) * 0.035;
    if x < panel.x0 - pad_x || x > panel.x1 + pad_x || y < panel.y0 - pad_y || y > panel.y1 + pad_y
    {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }
    let edge = ((x - panel.x0).abs())
        .min((x - panel.x1).abs())
        .min((y - panel.y0).abs())
        .min((y - panel.y1).abs());
    let border = if edge.abs() < 0.0025 { 0.18 } else { 0.0 };
    AnalyzerLight {
        alpha: 1.0,
        color: Color::new(
            1.0 + 16.0 * border,
            1.0 + 12.0 * border,
            2.0 + 20.0 * border,
        ),
    }
}

fn waveform_black_box(x: f32, y: f32, panel: PanelRect) -> AnalyzerLight {
    let pad_x = (panel.x1 - panel.x0) * 0.018;
    let pad_y = (panel.y1 - panel.y0) * 0.020;
    if x < panel.x0 - pad_x || x > panel.x1 + pad_x || y < panel.y0 - pad_y || y > panel.y1 + pad_y
    {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }
    let edge = ((x - panel.x0).abs())
        .min((x - panel.x1).abs())
        .min((y - panel.y0).abs())
        .min((y - panel.y1).abs());
    let border = if edge.abs() < 0.0022 { 0.15 } else { 0.0 };
    AnalyzerLight {
        alpha: 1.0,
        color: Color::new(
            2.0 + 18.0 * border,
            1.0 + 12.0 * border,
            3.0 + 22.0 * border,
        ),
    }
}

fn existing_electric_intensity(
    source_x: f32,
    source_y: f32,
    color: Color,
    t: f64,
    audio: AudioFeature,
) -> AnalyzerLight {
    let brightness = (color.r + color.g + color.b) / 3.0;
    let purple_blue = ((color.r + color.b) * 0.5 - color.g * 0.72).max(0.0) / 255.0;
    let warm_arc = ((color.r * 0.9 + color.g * 0.34) - color.b * 0.48).max(0.0) / 255.0;
    let bright_edge = smoothstep(72.0, 205.0, brightness);
    let top_electric_region: f32 = if source_y < 0.34 { 1.0 } else { 0.0 };
    let tower_region: f32 = if source_x > 0.42 && source_x < 0.58 && source_y < 0.64 {
        1.0
    } else {
        0.0
    };
    let lightning_region: f32 = if source_y < 0.52 && (purple_blue > 0.16 || warm_arc > 0.18) {
        1.0
    } else {
        0.0
    };
    let halo_region: f32 = if source_x > 0.34 && source_x < 0.66 && source_y < 0.28 {
        1.0
    } else {
        0.0
    };
    let region = top_electric_region
        .max(tower_region)
        .max(lightning_region)
        .max(halo_region);
    let electric_mask =
        (bright_edge * (purple_blue * 0.82 + warm_arc * 0.55) * region).clamp(0.0, 1.0);
    if electric_mask <= 0.0 {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }

    let high_flicker = 0.70
        + 0.30
            * (t as f32 * (8.5 + 4.0 * audio.high) + source_x * 33.0)
                .sin()
                .abs();
    let bass_core = 0.72 + 0.24 * audio.bass + 0.18 * audio.beat;
    let mid_halo = 0.74 + 0.24 * audio.rms + 0.14 * (t as f32 * 1.6).sin().max(0.0);
    let intensity = (electric_mask * high_flicker * bass_core * mid_halo).clamp(0.0, 0.72);
    AnalyzerLight {
        alpha: intensity,
        color: Color::new(
            78.0 + 92.0 * audio.high,
            42.0 + 58.0 * audio.rms,
            102.0 + 86.0 * audio.beat,
        ),
    }
}

fn letterbox_status_light(
    x: f32,
    y: f32,
    content: PanelRect,
    t: f64,
    beat: f32,
    duration_label: &str,
) -> AnalyzerLight {
    let mut alpha = 0.0_f32;
    if x < content.x0 {
        let w = content.x0.max(0.0001);
        let px = (x / w).clamp(0.0, 1.0);
        let py = y.clamp(0.0, 1.0);
        alpha = alpha
            .max(block_text_alpha(px, py, "TRUEVISION", 0.12, 0.08, 0.0042))
            .max(block_text_alpha(
                px,
                py,
                "COPYRIGHT 2026",
                0.08,
                0.17,
                0.0030,
            ))
            .max(block_text_alpha(px, py, "PROTOTYPE", 0.12, 0.25, 0.0038))
            .max(block_text_alpha(px, py, duration_label, 0.12, 0.36, 0.0037))
            .max(block_text_alpha(px, py, "CPU RAM LOG", 0.12, 0.45, 0.0033));
    } else if x > content.x1 {
        let w = (1.0 - content.x1).max(0.0001);
        let px = ((x - content.x1) / w).clamp(0.0, 1.0);
        let py = y.clamp(0.0, 1.0);
        alpha = alpha
            .max(block_text_alpha(px, py, "TRUEVISION", 0.10, 0.08, 0.0042))
            .max(block_text_alpha(px, py, "MANIFEST", 0.12, 0.18, 0.0038))
            .max(block_text_alpha(px, py, "CPU RAM", 0.14, 0.27, 0.0038))
            .max(block_text_alpha(px, py, "TIME LOG", 0.14, 0.36, 0.0038))
            .max(block_text_alpha(px, py, "YEAH", 0.22, 0.48, 0.0048));
    }
    let pulse = 0.58 + 0.18 * beat + 0.08 * (t as f32 * 1.4).sin().max(0.0);
    AnalyzerLight {
        alpha: (alpha * pulse).clamp(0.0, 0.70),
        color: Color::new(118.0, 72.0, 154.0),
    }
}

fn analyzer_panel_light(
    x: f32,
    y: f32,
    panel: PanelRect,
    bars: usize,
    t: f64,
    rms: f32,
    bass: f32,
    high: f32,
    beat: f32,
) -> AnalyzerLight {
    if x < panel.x0 || x > panel.x1 || y < panel.y0 || y > panel.y1 {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }
    let px = ((x - panel.x0) / (panel.x1 - panel.x0).max(0.0001)).clamp(0.0, 1.0);
    let py = ((y - panel.y0) / (panel.y1 - panel.y0).max(0.0001)).clamp(0.0, 1.0);
    let bar_pos = px * bars as f32;
    let bar = bar_pos.floor();
    let within = bar_pos.fract();
    if within < 0.11 || within > 0.89 {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }

    let band = (bar / (bars.saturating_sub(1).max(1)) as f32).clamp(0.0, 1.0);
    let low = (1.0 - smoothstep(0.02, 0.35, band)).clamp(0.0, 1.0);
    let mid = (1.0 - ((band - 0.50).abs() / 0.30)).clamp(0.0, 1.0);
    let treble = smoothstep(0.58, 0.98, band);
    let lag = hash2(bar, 44.4);
    let inertia = 0.15 * (t as f32 * (0.82 + lag * 0.55) + bar * 0.21).sin().max(0.0);
    let drive = (0.10
        + 0.92 * bass * low
        + 0.74 * rms * mid
        + 0.88 * high * treble
        + 0.46 * beat
        + inertia)
        .clamp(0.06, 1.0);

    let rise = (1.0 - py).clamp(0.0, 1.0);
    let active_height = (0.10 + 0.86 * drive).clamp(0.08, 0.98);
    let decay_tail = 0.04 + 0.08 * beat + 0.03 * lag;
    if rise > active_height + decay_tail {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }

    let led_row = ((rise * 18.0).fract() - 0.5).abs();
    let led_shape = 1.0 - smoothstep(0.24, 0.49, led_row);
    let column_shape = 1.0 - smoothstep(0.32, 0.50, (within - 0.5).abs());
    let decay = 1.0 - smoothstep(active_height, active_height + decay_tail, rise);
    let alpha = (led_shape * column_shape * decay * (0.16 + 0.88 * drive)).clamp(0.0, 1.0);
    AnalyzerLight {
        alpha,
        color: spectrum_band_color(band, drive, beat),
    }
}

fn vocal_presence(audio: AudioFeature) -> f32 {
    (0.82 * audio.rms + 0.92 * audio.high - 0.34 * audio.bass + 0.12 * audio.beat).clamp(0.0, 1.0)
}

fn panel_uv(x: f32, y: f32, panel: PanelRect) -> Option<(f32, f32)> {
    if x < panel.x0 || x > panel.x1 || y < panel.y0 || y > panel.y1 {
        None
    } else {
        Some((
            ((x - panel.x0) / (panel.x1 - panel.x0).max(0.0001)).clamp(0.0, 1.0),
            ((y - panel.y0) / (panel.y1 - panel.y0).max(0.0001)).clamp(0.0, 1.0),
        ))
    }
}

fn photo_schema_purple(strength: f32) -> Color {
    let s = strength.clamp(0.0, 1.35);
    Color::new(96.0 + 82.0 * s, 34.0 + 58.0 * s, 92.0 + 118.0 * s)
}

fn photo_schema_blue(strength: f32) -> Color {
    let s = strength.clamp(0.0, 1.35);
    Color::new(112.0 + 72.0 * s, 54.0 + 64.0 * s, 76.0 + 88.0 * s)
}

fn vocal_wave_panel_light(x: f32, y: f32, panel: PanelRect, audio: AudioFeature) -> AnalyzerLight {
    let Some((px, py)) = panel_uv(x, y, panel) else {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    };
    let wave_pos = px * (WAVEFORM_BINS - 1) as f32;
    let left = wave_pos.floor() as usize;
    let right = (left + 1).min(WAVEFORM_BINS - 1);
    let mix = wave_pos.fract();
    let wave = audio.waveform[left] * (1.0 - mix) + audio.waveform[right] * mix;
    let vocal = vocal_presence(audio);
    let center = 0.50 - wave * (0.33 + 0.05 * vocal);
    let distance = (py - center).abs();
    let trace = 1.0 - smoothstep(0.0055, 0.022 + 0.010 * vocal, distance);
    let glow = 1.0 - smoothstep(0.020, 0.075 + 0.020 * audio.beat, distance);
    let centerline = 1.0 - smoothstep(0.0015, 0.006, (py - 0.50).abs());
    let graticule = waveform_graticule(px, py);
    let gate = (0.58 + 0.50 * vocal + 0.24 * audio.beat).clamp(0.0, 1.0);
    let alpha = ((trace * 1.12 + glow * 0.26 + centerline * 0.08 + graticule * 0.10) * gate)
        .clamp(0.0, 1.0);
    AnalyzerLight {
        alpha,
        color: waveform_color(py, audio),
    }
}

fn waveform_graticule(px: f32, py: f32) -> f32 {
    let vertical = ((px * 12.0).fract() - 0.5).abs();
    let horizontal = ((py * 4.0).fract() - 0.5).abs();
    let v = 1.0 - smoothstep(0.012, 0.030, vertical);
    let h = 1.0 - smoothstep(0.010, 0.026, horizontal);
    (v.max(h) * 0.55).clamp(0.0, 1.0)
}

fn waveform_color(py: f32, audio: AudioFeature) -> Color {
    let energy = (0.52 * audio.rms + 0.28 * audio.high + 0.20 * audio.beat).clamp(0.0, 1.0);
    let blue = Color::new(218.0, 124.0, 34.0);
    let violet = Color::new(168.0, 88.0, 188.0);
    let gold = Color::new(36.0, 170.0, 232.0);
    let t = smoothstep(0.14, 0.86, py);
    let base = if t < 0.70 {
        lerp_color(blue, violet, t / 0.70)
    } else {
        lerp_color(violet, gold, (t - 0.70) / 0.30)
    };
    let pulse = 0.92 + 0.52 * energy;
    Color::new(base.b * pulse, base.g * pulse, base.r * pulse)
}

fn headphone_soundfield_light(
    x: f32,
    y: f32,
    panel: PanelRect,
    t: f64,
    rms: f32,
    bass: f32,
    high: f32,
    beat: f32,
) -> AnalyzerLight {
    let Some((px, py)) = panel_uv(x, y, panel) else {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    };
    let dx = (px - 0.50) * 1.15;
    let dy = (py - 0.50) * 1.00;
    let radius = (dx * dx + dy * dy).sqrt();
    if radius > 0.54 {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    }

    let angle = dy.atan2(dx);
    let sweep_angle = t as f32 * (0.55 + 0.28 * beat) + 1.7 * bass;
    let mut angle_delta = (angle - sweep_angle).abs();
    while angle_delta > std::f32::consts::PI {
        angle_delta -= std::f32::consts::TAU;
    }
    angle_delta = angle_delta.abs();
    let sweep = (1.0 - smoothstep(0.018, 0.20, angle_delta)) * (0.16 + 0.52 * rms + 0.22 * beat);
    let ring_a = 1.0 - smoothstep(0.008, 0.028, (radius - (0.18 + 0.07 * bass)).abs());
    let ring_b = 1.0 - smoothstep(0.008, 0.026, (radius - (0.34 + 0.05 * high)).abs());
    let ring_c = 1.0 - smoothstep(0.008, 0.024, (radius - 0.48).abs());
    let left_lobe = lobe_strength(px, py, 0.35 - 0.05 * bass, 0.50 + 0.05 * high);
    let right_lobe = lobe_strength(px, py, 0.65 + 0.05 * bass, 0.50 - 0.05 * high);
    let center_dot = 1.0 - smoothstep(0.018, 0.055, radius);
    let alpha = (sweep
        + 0.22 * ring_a * (0.35 + bass)
        + 0.18 * ring_b * (0.35 + high)
        + 0.14 * ring_c
        + 0.26 * (left_lobe + right_lobe) * (0.30 + rms)
        + 0.20 * center_dot)
        .clamp(0.0, 0.78);
    AnalyzerLight {
        alpha,
        color: if high > bass {
            photo_schema_blue(0.70 + high)
        } else {
            photo_schema_purple(0.68 + bass)
        },
    }
}

fn lobe_strength(x: f32, y: f32, cx: f32, cy: f32) -> f32 {
    let dx = (x - cx) / 0.115;
    let dy = (y - cy) / 0.085;
    (1.0 - smoothstep(0.18, 1.0, (dx * dx + dy * dy).sqrt())).clamp(0.0, 1.0)
}

fn prototype_mark_light(x: f32, y: f32, panel: PanelRect, t: f64, beat: f32) -> AnalyzerLight {
    let Some((px, py)) = panel_uv(x, y, panel) else {
        return AnalyzerLight {
            alpha: 0.0,
            color: Color::default(),
        };
    };
    let pulse = 0.72 + 0.18 * beat + 0.06 * (t as f32 * 2.0).sin().max(0.0);
    let alpha = block_text_alpha(px, py, "TRUEVISION", 0.02, 0.08, 0.0045)
        .max(block_text_alpha(
            px,
            py,
            "COPYRIGHT 2026",
            0.02,
            0.40,
            0.0035,
        ))
        .max(block_text_alpha(px, py, "PROTOTYPE", 0.02, 0.68, 0.0045));
    AnalyzerLight {
        alpha: (alpha * pulse).clamp(0.0, 0.72),
        color: Color::new(132.0, 78.0, 154.0),
    }
}

fn block_text_alpha(x: f32, y: f32, text: &str, x0: f32, y0: f32, scale: f32) -> f32 {
    let cell_x = scale * 0.55;
    let cell_y = scale;
    let gx = ((x - x0) / cell_x).floor() as i32;
    let gy = ((y - y0) / cell_y).floor() as i32;
    if gy < 0 || gy >= 7 || gx < 0 {
        return 0.0;
    }
    let cell_w = 6_i32;
    let char_index = gx / cell_w;
    let col = gx % cell_w;
    if col >= 5 || char_index < 0 || char_index as usize >= text.chars().count() {
        return 0.0;
    }
    let ch = text.chars().nth(char_index as usize).unwrap_or(' ');
    let glyph = glyph_5x7(ch);
    let row = glyph[gy as usize].as_bytes();
    if row[col as usize] == b'1' {
        1.0
    } else {
        0.0
    }
}

fn glyph_5x7(ch: char) -> [&'static str; 7] {
    match ch {
        'A' => [
            "01110", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'C' => [
            "01111", "10000", "10000", "10000", "10000", "10000", "01111",
        ],
        'D' => [
            "11110", "10001", "10001", "10001", "10001", "10001", "11110",
        ],
        'E' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "11111",
        ],
        'F' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "10000",
        ],
        'G' => [
            "01111", "10000", "10000", "10011", "10001", "10001", "01111",
        ],
        'H' => [
            "10001", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'I' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "11111",
        ],
        'L' => [
            "10000", "10000", "10000", "10000", "10000", "10000", "11111",
        ],
        'M' => [
            "10001", "11011", "10101", "10101", "10001", "10001", "10001",
        ],
        'N' => [
            "10001", "11001", "10101", "10011", "10001", "10001", "10001",
        ],
        'O' => [
            "01110", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'P' => [
            "11110", "10001", "10001", "11110", "10000", "10000", "10000",
        ],
        'R' => [
            "11110", "10001", "10001", "11110", "10100", "10010", "10001",
        ],
        'S' => [
            "01111", "10000", "10000", "01110", "00001", "00001", "11110",
        ],
        'T' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "00100",
        ],
        'U' => [
            "10001", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'V' => [
            "10001", "10001", "10001", "10001", "10001", "01010", "00100",
        ],
        'Y' => [
            "10001", "10001", "01010", "00100", "00100", "00100", "00100",
        ],
        '1' => [
            "00100", "01100", "00100", "00100", "00100", "00100", "01110",
        ],
        '0' => [
            "01110", "10001", "10011", "10101", "11001", "10001", "01110",
        ],
        '2' => [
            "01110", "10001", "00001", "00010", "00100", "01000", "11111",
        ],
        '3' => [
            "11110", "00001", "00001", "01110", "00001", "00001", "11110",
        ],
        '4' => [
            "10010", "10010", "10010", "11111", "00010", "00010", "00010",
        ],
        '5' => [
            "11111", "10000", "10000", "11110", "00001", "00001", "11110",
        ],
        '6' => [
            "00110", "01000", "10000", "11110", "10001", "10001", "01110",
        ],
        '7' => [
            "11111", "00001", "00010", "00100", "01000", "01000", "01000",
        ],
        '8' => [
            "01110", "10001", "10001", "01110", "10001", "10001", "01110",
        ],
        '9' => [
            "01110", "10001", "10001", "01111", "00001", "00010", "11100",
        ],
        ' ' => [
            "00000", "00000", "00000", "00000", "00000", "00000", "00000",
        ],
        _ => [
            "00000", "00000", "00000", "00000", "00000", "00000", "00000",
        ],
    }
}

fn spectrum_backdrop_haze(x: f32, y: f32, t: f64, rms: f32, high: f32) -> f32 {
    let noise = value_noise(x * 7.0 + t as f32 * 0.05, y * 5.0 - t as f32 * 0.025);
    let center = (1.0 - (x - 0.5).abs() * 1.7).clamp(0.0, 1.0);
    (noise * center * smoothstep(0.18, 0.92, y) * (0.035 + 0.10 * rms + 0.06 * high))
        .clamp(0.0, 0.22)
}

fn spectrum_band_color(band: f32, drive: f32, beat: f32) -> Color {
    let pulse = (0.70 + 0.34 * drive + 0.28 * beat).clamp(0.0, 1.35);
    let deep_blue = Color::new(190.0, 78.0, 16.0);
    let electric_blue = Color::new(236.0, 148.0, 34.0);
    let warm_gold = Color::new(30.0, 174.0, 238.0);
    let t = band.clamp(0.0, 1.0);
    let base = if t < 0.55 {
        lerp_color(deep_blue, electric_blue, t / 0.55)
    } else {
        lerp_color(electric_blue, warm_gold, (t - 0.55) / 0.45)
    };
    Color::new(base.b * pulse, base.g * pulse, base.r * pulse)
}

fn render_lyric_city_frame(
    args: &Args,
    time_seconds: f64,
    frame_index: usize,
    audio: AudioFeature,
    frame: &mut [u8],
    stats: &mut FrameStats,
) -> String {
    let width = args.width;
    let height = args.height;
    let norm = (time_seconds / args.duration.max(0.001)).clamp(0.0, 1.0) as f32;
    let phase = lyric_phase(norm);
    let phase_pressure = phase.fire_pressure;
    let fire = (phase_pressure + 0.34 * audio.bass + 0.26 * audio.beat).clamp(0.0, 1.0);
    let memory = phase.memory_pressure;
    let defiance = (phase.defiance_pressure + 0.30 * audio.rms).clamp(0.0, 1.0);
    let phoenix = (phase.phoenix_pressure + 0.36 * audio.high + 0.20 * audio.beat).clamp(0.0, 1.0);
    let fog_push = (0.18 + 0.34 * memory + 0.24 * audio.rms).clamp(0.0, 0.82);
    let mut fog_accum = 0.0_f64;
    let mut glow_pixels = 0_u64;
    let mut occluded_pixels = 0_u64;
    let camera_scale = 1.12 + 0.035 * (time_seconds as f32 * 0.045).sin();
    let camera_pan_x = 0.026 * (time_seconds as f32 * 0.038).sin();
    let camera_pan_y = -0.028 + 0.012 * (time_seconds as f32 * 0.031).sin();
    let palette = args.palette.as_str();

    for y in 0..height {
        let yf = y as f32 / height as f32;
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let sx = 0.5 + (xf - 0.5) * camera_scale + camera_pan_x;
            let sy = 0.5 + (yf - 0.5) * camera_scale + camera_pan_y;
            let vx = ((sx.clamp(0.0, 0.999) * width as f32) as usize).min(width - 1);
            let vy = ((sy.clamp(0.0, 0.999) * height as f32) as usize).min(height - 1);
            let mut color = lyric_city_background(sx, sy, time_seconds, memory, defiance, palette);

            let stars = lyric_star_field(sx, sy, time_seconds, audio.high, audio.beat);
            if stars > 0.0 {
                blend_add(&mut color, palette_star_color(palette), stars);
                glow_pixels += 1;
            }

            let moon = lyric_full_moon(sx, sy, time_seconds, audio.beat);
            if moon > 0.0 {
                blend_add(&mut color, palette_moon_color(palette), moon);
                glow_pixels += 1;
            }

            let skyline_top = skyline_top(sx, time_seconds, width, height);
            if (y as f32) >= skyline_top {
                let building_alpha = (1.0
                    - smoothstep(skyline_top, skyline_top + height as f32 * 0.055, y as f32))
                .max(0.0);
                blend(
                    &mut color,
                    palette_building_color(palette, memory, defiance),
                    building_alpha.max(0.94),
                );
                let window = city_window_light(
                    vx,
                    vy,
                    time_seconds,
                    fire,
                    phase.window_pressure,
                    audio.beat,
                    audio.bass,
                );
                if window > 0.0 {
                    blend_add(
                        &mut color,
                        palette_window_color(palette, sx, sy, time_seconds, fire, audio.beat),
                        window,
                    );
                    glow_pixels += 1;
                }
                let spectrum = skyline_spectrum_light(
                    sx,
                    sy,
                    y as f32,
                    skyline_top,
                    height as f32,
                    time_seconds,
                    fire,
                    audio.beat,
                    audio.bass,
                    audio.high,
                );
                if spectrum > 0.0 {
                    blend_add(
                        &mut color,
                        palette_spectrum_color(
                            palette,
                            sx,
                            time_seconds,
                            fire,
                            audio.beat,
                            audio.bass,
                        ),
                        spectrum,
                    );
                    glow_pixels += 1;
                }
                occluded_pixels += 1;
            }

            let horizon_glow = horizon_fire_glow(sx, sy, time_seconds, fire, defiance);
            if horizon_glow > 0.0 {
                glow_pixels += 1;
                blend_add(
                    &mut color,
                    palette_horizon_color(palette, fire, defiance),
                    horizon_glow,
                );
            }

            apply_wet_city_reflection(&mut color, sx, sy, time_seconds, fire, audio.bass, palette);

            let ash = lyric_ash_field(sx, sy, time_seconds, phoenix, fire, audio.high);
            if ash > 0.0 {
                blend_add(&mut color, palette_ash_color(palette, fire, phoenix), ash);
            }

            let silhouette = lyric_pair_silhouette_alpha(sx, sy, norm, time_seconds);
            if silhouette > 0.0 {
                blend(&mut color, palette_silhouette_color(palette), silhouette);
                let rim = silhouette * (0.18 + 0.33 * fire + 0.22 * phoenix);
                blend_add(&mut color, palette_rim_color(palette, fire, phoenix), rim);
                occluded_pixels += 1;
            }

            let fog = lyric_city_fog(sx, sy, time_seconds, fog_push, memory, phoenix);
            fog_accum += fog as f64;
            blend(&mut color, palette_fog_color(palette, memory, fire), fog);

            write_pixel(frame, width, x, y, color);
        }
    }

    stats.fog_coverage_sum += fog_accum / (width * height) as f64;
    stats.fog_samples += 1;
    stats.occluded_pixels_sum += occluded_pixels;
    stats.glow_pixels_sum += glow_pixels;

    format!(
        "{{\"frame_index\":{},\"time_seconds\":{:.6},\"scene\":\"lyric_city_silhouette\",\"palette\":\"{}\",\"lyric_phase\":\"{}\",\"audio\":{{\"rms\":{:.6},\"bass\":{:.6},\"high\":{:.6},\"beat\":{:.6}}},\"camera\":{{\"scale\":{:.6},\"pan_x\":{:.6},\"pan_y\":{:.6}}},\"fog_mean\":{:.6},\"occluded_pixels\":{},\"glow_pixels\":{},\"state_layers\":[\"wide_backed_up_camera_pan\",\"full_moon_night_sky\",\"deterministic_star_field\",\"black_skyline_band_45_60_height\",\"beat_banged_window_lights\",\"memory_fog\",\"bass_horizon_fire\",\"wet_reflections\",\"father_child_silhouette_arc\",\"phoenix_ash_rise\",\"lyric_phase_color_pressure\"]}}",
        frame_index,
        time_seconds,
        palette,
        phase.name,
        audio.rms,
        audio.bass,
        audio.high,
        audio.beat,
        camera_scale,
        camera_pan_x,
        camera_pan_y,
        fog_accum / (width * height) as f64,
        occluded_pixels,
        glow_pixels
    )
}

struct LyricPhase {
    name: &'static str,
    fire_pressure: f32,
    memory_pressure: f32,
    defiance_pressure: f32,
    phoenix_pressure: f32,
    window_pressure: f32,
}

fn lyric_phase(norm: f32) -> LyricPhase {
    if norm < 0.07 {
        LyricPhase {
            name: "spoken_intro_baby_memory",
            fire_pressure: 0.10,
            memory_pressure: 0.95,
            defiance_pressure: 0.10,
            phoenix_pressure: 0.00,
            window_pressure: 0.18,
        }
    } else if norm < 0.22 {
        LyricPhase {
            name: "verse_one_separation",
            fire_pressure: 0.22,
            memory_pressure: 0.80,
            defiance_pressure: 0.24,
            phoenix_pressure: 0.00,
            window_pressure: 0.25,
        }
    } else if norm < 0.31 {
        LyricPhase {
            name: "prechorus_truth_rising",
            fire_pressure: 0.42,
            memory_pressure: 0.58,
            defiance_pressure: 0.44,
            phoenix_pressure: 0.08,
            window_pressure: 0.40,
        }
    } else if norm < 0.45 {
        LyricPhase {
            name: "chorus_world_on_fire",
            fire_pressure: 0.86,
            memory_pressure: 0.25,
            defiance_pressure: 0.82,
            phoenix_pressure: 0.12,
            window_pressure: 0.72,
        }
    } else if norm < 0.57 {
        LyricPhase {
            name: "verse_two_blood_and_superman",
            fire_pressure: 0.38,
            memory_pressure: 0.58,
            defiance_pressure: 0.62,
            phoenix_pressure: 0.10,
            window_pressure: 0.44,
        }
    } else if norm < 0.67 {
        LyricPhase {
            name: "build_no_dad_hero_zero",
            fire_pressure: 0.48,
            memory_pressure: 0.82,
            defiance_pressure: 0.52,
            phoenix_pressure: 0.20,
            window_pressure: 0.38,
        }
    } else if norm < 0.79 {
        LyricPhase {
            name: "drop_rescue_defiance",
            fire_pressure: 0.92,
            memory_pressure: 0.25,
            defiance_pressure: 0.96,
            phoenix_pressure: 0.34,
            window_pressure: 0.85,
        }
    } else if norm < 0.88 {
        LyricPhase {
            name: "bridge_phoenix_ashes",
            fire_pressure: 0.72,
            memory_pressure: 0.45,
            defiance_pressure: 0.62,
            phoenix_pressure: 1.00,
            window_pressure: 0.62,
        }
    } else if norm < 0.97 {
        LyricPhase {
            name: "final_chorus_funeral_pyre",
            fire_pressure: 1.00,
            memory_pressure: 0.22,
            defiance_pressure: 1.00,
            phoenix_pressure: 0.72,
            window_pressure: 0.95,
        }
    } else {
        LyricPhase {
            name: "outro_baby_memory_return",
            fire_pressure: 0.18,
            memory_pressure: 0.95,
            defiance_pressure: 0.22,
            phoenix_pressure: 0.10,
            window_pressure: 0.18,
        }
    }
}

fn lyric_city_background(
    x: f32,
    y: f32,
    t: f64,
    memory: f32,
    defiance: f32,
    palette: &str,
) -> Color {
    let storm = 0.5 + 0.5 * (x * 8.0 + y * 4.0 + t as f32 * 0.055).sin();
    let cold = if is_glitch_palette(palette) {
        Color::new(
            30.0 + 18.0 * memory,
            8.0 + 10.0 * memory,
            18.0 + 14.0 * memory,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            20.0 + 16.0 * memory,
            9.0 + 11.0 * memory,
            20.0 + 18.0 * memory,
        )
    } else {
        Color::new(
            10.0 + 13.0 * memory,
            11.0 + 13.0 * memory,
            20.0 + 22.0 * memory,
        )
    };
    let angry = if is_glitch_palette(palette) {
        Color::new(
            40.0 + 34.0 * defiance,
            4.0 + 22.0 * defiance,
            34.0 + 28.0 * defiance,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            30.0 + 22.0 * defiance,
            10.0 + 20.0 * defiance,
            28.0 + 16.0 * defiance,
        )
    } else {
        Color::new(5.0, 9.0 + 10.0 * defiance, 16.0 + 28.0 * defiance)
    };
    let base = lerp_color(cold, angry, defiance * 0.68);
    let vertical = (1.0 - y).clamp(0.0, 1.2);
    if is_glitch_palette(palette) {
        Color::new(
            base.b + 42.0 * vertical + 12.0 * storm,
            base.g + 9.0 * vertical + 6.0 * storm,
            base.r + 24.0 * vertical + 7.0 * storm,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            base.b + 34.0 * vertical + 8.0 * storm,
            base.g + 12.0 * vertical + 3.0 * storm,
            base.r + 20.0 * vertical + 4.0 * storm,
        )
    } else {
        Color::new(
            base.b + 22.0 * vertical + 6.0 * storm,
            base.g + 17.0 * vertical,
            base.r + 15.0 * vertical,
        )
    }
}

fn is_unity_palette(palette: &str) -> bool {
    matches!(palette, "unity_signal" | "middle_finger_to_racism")
}

fn is_glitch_palette(palette: &str) -> bool {
    matches!(palette, "glitch_444" | "glitch")
}

fn palette_star_color(palette: &str) -> Color {
    if is_glitch_palette(palette) {
        Color::new(205.0, 238.0, 255.0)
    } else if is_unity_palette(palette) {
        Color::new(170.0, 214.0, 235.0)
    } else {
        Color::new(118.0, 138.0, 166.0)
    }
}

fn palette_moon_color(palette: &str) -> Color {
    if is_glitch_palette(palette) {
        Color::new(212.0, 232.0, 255.0)
    } else if is_unity_palette(palette) {
        Color::new(186.0, 222.0, 240.0)
    } else {
        Color::new(174.0, 194.0, 220.0)
    }
}

fn palette_building_color(palette: &str, memory: f32, defiance: f32) -> Color {
    if is_glitch_palette(palette) {
        Color::new(2.0 + 2.0 * memory, 1.0 + 1.5 * defiance, 3.0 + 2.0 * memory)
    } else if is_unity_palette(palette) {
        Color::new(2.0 + 1.5 * memory, 1.5 + 2.5 * defiance, 4.0 + 1.5 * memory)
    } else {
        Color::new(1.0, 2.0 + 1.5 * defiance, 4.0 + 2.0 * memory)
    }
}

fn palette_signal_color(palette: &str, selector: f32, fire: f32, beat: f32) -> Color {
    if is_glitch_palette(palette) {
        let band = (selector * 7.0).floor() as i32;
        let pulse = 0.72 + 0.52 * beat + 0.18 * fire;
        return match band.rem_euclid(7) {
            0 => Color::new(240.0 * pulse, 255.0 * pulse, 250.0 * pulse),
            1 => Color::new(250.0 * pulse, 92.0 * pulse, 214.0 * pulse),
            2 => Color::new(82.0 * pulse, 255.0 * pulse, 92.0 * pulse),
            3 => Color::new(255.0 * pulse, 198.0 * pulse, 54.0 * pulse),
            4 => Color::new(255.0 * pulse, 72.0 * pulse, 82.0 * pulse),
            5 => Color::new(118.0 * pulse, 76.0 * pulse, 255.0 * pulse),
            _ => Color::new(42.0 * pulse, 248.0 * pulse, 255.0 * pulse),
        };
    }
    if !is_unity_palette(palette) {
        return Color::new(8.0, 48.0 + 50.0 * fire, 94.0 + 102.0 * fire);
    }
    let band = (selector * 6.0).floor() as i32;
    let pulse = 0.7 + 0.45 * beat + 0.25 * fire;
    match band.rem_euclid(6) {
        0 => Color::new(185.0 * pulse, 220.0 * pulse, 42.0 * pulse),
        1 => Color::new(54.0 * pulse, 220.0 * pulse, 92.0 * pulse),
        2 => Color::new(220.0 * pulse, 82.0 * pulse, 220.0 * pulse),
        3 => Color::new(246.0 * pulse, 236.0 * pulse, 228.0 * pulse),
        4 => Color::new(78.0 * pulse, 176.0 * pulse, 244.0 * pulse),
        _ => Color::new(172.0 * pulse, 72.0 * pulse, 244.0 * pulse),
    }
}

fn palette_window_color(palette: &str, x: f32, y: f32, t: f64, fire: f32, beat: f32) -> Color {
    let selector = (x * 3.1 + y * 1.7 + t as f32 * 0.016).fract().abs();
    palette_signal_color(palette, selector, fire, beat)
}

fn palette_spectrum_color(palette: &str, x: f32, t: f64, fire: f32, beat: f32, bass: f32) -> Color {
    if is_unity_palette(palette) {
        let selector = (x * 4.8 + t as f32 * 0.024 + bass * 0.35).fract().abs();
        palette_signal_color(palette, selector, fire + 0.25 * bass, beat)
    } else {
        Color::new(4.0, 35.0 + 48.0 * fire, 76.0 + 88.0 * fire)
    }
}

fn palette_horizon_color(palette: &str, fire: f32, defiance: f32) -> Color {
    if is_glitch_palette(palette) {
        Color::new(
            84.0 + 94.0 * defiance,
            20.0 + 92.0 * fire,
            118.0 + 64.0 * fire,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            42.0 + 76.0 * defiance,
            42.0 + 80.0 * fire,
            92.0 + 38.0 * fire,
        )
    } else {
        Color::new(4.0, 62.0 + 58.0 * fire, 146.0 + 70.0 * fire)
    }
}

fn palette_ash_color(palette: &str, fire: f32, phoenix: f32) -> Color {
    if is_glitch_palette(palette) {
        Color::new(
            120.0 + 90.0 * phoenix,
            70.0 + 154.0 * phoenix,
            170.0 + 58.0 * fire,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            86.0 + 80.0 * phoenix,
            170.0 + 52.0 * phoenix,
            74.0 + 84.0 * fire,
        )
    } else {
        Color::new(9.0, 58.0 + 34.0 * phoenix, 156.0 + 58.0 * fire)
    }
}

fn palette_silhouette_color(_palette: &str) -> Color {
    Color::new(1.5, 2.0, 4.0)
}

fn palette_rim_color(palette: &str, fire: f32, phoenix: f32) -> Color {
    if is_glitch_palette(palette) {
        Color::new(
            58.0 + 82.0 * phoenix,
            172.0 + 36.0 * fire,
            150.0 + 70.0 * phoenix,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            42.0 + 34.0 * phoenix,
            118.0 + 42.0 * fire,
            116.0 + 42.0 * phoenix,
        )
    } else {
        Color::new(5.0, 38.0, 84.0 + 50.0 * fire)
    }
}

fn palette_fog_color(palette: &str, memory: f32, fire: f32) -> Color {
    if is_glitch_palette(palette) {
        Color::new(
            44.0 + 38.0 * memory,
            22.0 + 18.0 * memory,
            48.0 + 22.0 * fire,
        )
    } else if is_unity_palette(palette) {
        Color::new(
            36.0 + 30.0 * memory,
            30.0 + 24.0 * memory,
            48.0 + 12.0 * fire,
        )
    } else {
        Color::new(
            26.0 + 22.0 * memory,
            32.0 + 16.0 * memory,
            42.0 + 28.0 * memory + 18.0 * fire,
        )
    }
}

fn skyline_top(x: f32, t: f64, _width: usize, height: usize) -> f32 {
    let block = (x * 34.0).floor();
    let base = 0.47 + 0.13 * hash2(block, 7.0) + 0.025 * hash2(block * 1.7, 3.0);
    let antenna = if hash2(block, 19.0) > 0.90 {
        0.035 * (0.65 + 0.35 * (t as f32 * 0.2).sin())
    } else {
        0.0
    };
    height as f32 * (base - antenna).clamp(0.44, 0.60)
}

fn city_window_light(
    x: usize,
    y: usize,
    t: f64,
    fire: f32,
    pressure: f32,
    beat: f32,
    bass: f32,
) -> f32 {
    let wx = x / 16;
    let wy = y / 20;
    let gate = hash2(wx as f32, wy as f32);
    if gate < 0.81 {
        return 0.0;
    }
    let blink = 0.50 + 0.50 * (t as f32 * (0.62 + gate) + gate * 9.0).sin();
    let beat_bang = 0.38 + 1.55 * beat + 0.72 * bass;
    ((gate - 0.81) * 3.45 * blink * beat_bang * (0.30 + 0.58 * pressure + 0.25 * fire))
        .clamp(0.0, 1.0)
}

fn skyline_spectrum_light(
    x: f32,
    y: f32,
    pixel_y: f32,
    skyline_top: f32,
    height: f32,
    t: f64,
    fire: f32,
    beat: f32,
    bass: f32,
    high: f32,
) -> f32 {
    let block = (x * 34.0).floor();
    let within_block = (x * 34.0).fract();
    if within_block < 0.18 || within_block > 0.82 {
        return 0.0;
    }

    let cell_row = ((pixel_y - skyline_top) / 13.0).floor();
    let stack_gate = hash2(block, cell_row + 41.0);
    if stack_gate < 0.58 {
        return 0.0;
    }

    let building_depth =
        ((pixel_y - skyline_top) / (height - skyline_top).max(1.0)).clamp(0.0, 1.0);
    let from_street = 1.0 - building_depth;
    let lower_floor = smoothstep(0.44, 1.0, building_depth);
    let mid_floor = (1.0 - ((building_depth - 0.55).abs() / 0.32)).clamp(0.0, 1.0);
    let upper_floor = smoothstep(0.42, 0.0, building_depth);
    let tower_lag = hash2(block, 23.0);
    let tower_phase = (t as f32 * (0.58 + tower_lag * 0.50) + block * 0.37)
        .sin()
        .max(0.0);
    let floor_drive =
        0.18 + 0.58 * bass * lower_floor + 0.36 * fire * mid_floor + 0.44 * high * upper_floor;
    let beat_push = 0.18 + 0.72 * beat * (0.55 + 0.45 * tower_lag);
    let band_drive = (floor_drive + beat_push + 0.12 * tower_phase).clamp(0.06, 1.0);
    let active_height = (0.12 + 0.78 * band_drive).clamp(0.08, 0.94);
    let persistence_tail = 0.07 + 0.07 * beat + 0.04 * tower_lag;
    if from_street > active_height + persistence_tail {
        return 0.0;
    }

    let row_segment = ((y * 78.0).fract() - 0.5).abs();
    let segment_shape = 1.0 - smoothstep(0.22, 0.48, row_segment);
    let column_taper = 1.0 - smoothstep(0.30, 0.50, (within_block - 0.5).abs());
    let rising_body =
        1.0 - smoothstep(active_height, active_height + persistence_tail, from_street);
    let floor_bias =
        (0.48 + 0.34 * lower_floor + 0.18 * mid_floor + 0.20 * upper_floor).clamp(0.35, 1.0);
    (segment_shape
        * column_taper
        * rising_body
        * floor_bias
        * (0.10 + 0.62 * band_drive)
        * (0.45 + 0.55 * stack_gate))
        .clamp(0.0, 0.70)
}

fn lyric_full_moon(x: f32, y: f32, t: f64, beat: f32) -> f32 {
    let mx = 0.78 + 0.012 * (t as f32 * 0.023).sin();
    let my = 0.18 + 0.006 * (t as f32 * 0.017).cos();
    let dx = x - mx;
    let dy = (y - my) * 1.12;
    let d = (dx * dx + dy * dy).sqrt();
    let disk = 1.0 - smoothstep(0.048, 0.057, d);
    let halo = (1.0 - smoothstep(0.058, 0.22, d)) * (0.16 + 0.08 * beat);
    (disk * 0.72 + halo).clamp(0.0, 1.0)
}

fn lyric_star_field(x: f32, y: f32, t: f64, high: f32, beat: f32) -> f32 {
    if y > 0.56 {
        return 0.0;
    }
    let cell_x = (x * 230.0).floor();
    let cell_y = (y * 132.0).floor();
    let gate = hash2(cell_x, cell_y);
    if gate < 0.986 {
        return 0.0;
    }
    let lx = (x * 230.0).fract() - 0.5;
    let ly = (y * 132.0).fract() - 0.5;
    let dot = 1.0 - smoothstep(0.02, 0.42, (lx * lx + ly * ly).sqrt());
    let twinkle = 0.50 + 0.50 * (t as f32 * (0.52 + gate * 1.8) + gate * 17.0).sin();
    (dot * twinkle * (0.16 + 0.62 * high + 0.22 * beat)).clamp(0.0, 0.85)
}

fn horizon_fire_glow(x: f32, y: f32, t: f64, fire: f32, defiance: f32) -> f32 {
    let horizon = 0.58 + 0.018 * (t as f32 * 0.15).sin();
    let low_band = (1.0 - smoothstep(0.0, 0.20, (y - horizon).abs())).clamp(0.0, 1.0);
    let below_horizon = smoothstep(horizon - 0.04, horizon + 0.22, y);
    let smoke_breakup = 0.42 + 0.58 * value_noise(x * 12.0 + t as f32 * 0.08, y * 7.0);
    (low_band * below_horizon * smoke_breakup * (0.10 + 0.24 * fire) * (0.72 + 0.22 * defiance))
        .clamp(0.0, 0.34)
}

fn apply_wet_city_reflection(
    color: &mut Color,
    x: f32,
    y: f32,
    t: f64,
    fire: f32,
    bass: f32,
    palette: &str,
) {
    if y < 0.58 {
        return;
    }
    let center = (1.0 - (x - 0.5).abs() * 1.45).max(0.0);
    let ripple = 0.5 + 0.5 * (x * 38.0 + y * 24.0 + t as f32 * (2.4 + 1.8 * bass)).sin();
    let streak = smoothstep(0.58, 1.0, y) * center * ripple * (0.12 + 0.40 * fire + 0.28 * bass);
    let reflection = if is_glitch_palette(palette) {
        Color::new(
            78.0 + 108.0 * bass,
            42.0 + 126.0 * fire,
            132.0 + 74.0 * bass,
        )
    } else if is_unity_palette(palette) {
        Color::new(36.0 + 92.0 * bass, 88.0 + 104.0 * fire, 96.0 + 42.0 * fire)
    } else {
        Color::new(8.0, 48.0 + 44.0 * fire, 110.0 + 72.0 * fire)
    };
    blend_add(color, reflection, streak);
}

fn lyric_ash_field(x: f32, y: f32, t: f64, phoenix: f32, fire: f32, high: f32) -> f32 {
    let rise = (y + t as f32 * (0.05 + 0.05 * phoenix)) % 1.0;
    let filament = (x * 64.0 + rise * 15.0 + t as f32 * 0.7).sin();
    let speck = value_noise(x * 92.0 + t as f32 * 2.2, rise * 74.0 - t as f32 * 0.9);
    let threshold = 0.84 - 0.18 * phoenix - 0.08 * high;
    if filament > threshold && speck > 0.70 {
        ((filament - threshold) * 2.7 * (0.35 + fire + phoenix + high)).clamp(0.0, 0.85)
    } else {
        0.0
    }
}

fn lyric_pair_silhouette_alpha(x: f32, y: f32, norm: f32, t: f64) -> f32 {
    let together = smoothstep(0.64, 0.92, norm);
    let child_x = 0.34 + 0.08 * together;
    let adult_x = 0.58 - 0.08 * together + 0.01 * (t as f32 * 0.2).sin();
    let child = human_silhouette(x, y, child_x, 0.72, 0.58);
    let adult = human_silhouette(x, y, adult_x, 0.70, 0.86);
    child.max(adult)
}

fn human_silhouette(x: f32, y: f32, cx: f32, foot_y: f32, scale: f32) -> f32 {
    let body_h = 0.24 * scale;
    let body_w = 0.034 * scale;
    let body_y = foot_y - body_h * 0.47;
    let nx = (x - cx) / body_w.max(0.001);
    let ny = (y - body_y) / body_h.max(0.001);
    let torso = 1.0 - nx * nx * 0.7 - ny * ny * 1.8;
    let head = 1.0
        - ((x - cx) / (body_w * 0.72)).powf(2.0)
        - ((y - (body_y - body_h * 0.62)) / (body_w * 0.82)).powf(2.0);
    let legs = if y > body_y + body_h * 0.15 && y < foot_y {
        let leg_spread = ((x - cx).abs() - body_w * 0.28).abs();
        (1.0 - leg_spread / (body_w * 0.34)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    torso.max(head).max(legs).clamp(0.0, 1.0)
}

fn lyric_city_fog(x: f32, y: f32, t: f64, fog_push: f32, memory: f32, phoenix: f32) -> f32 {
    let low = smoothstep(0.34, 0.96, y);
    let cloud = value_noise(x * 7.0 + t as f32 * 0.05, y * 5.0 - t as f32 * 0.025);
    let bands = 0.5 + 0.5 * (x * 9.0 - y * 6.0 + t as f32 * (0.20 + 0.18 * phoenix)).sin();
    (low * (cloud * 0.45 + bands * 0.30) * (fog_push + 0.16 * memory)).clamp(0.0, 0.68)
}

fn render_warp_laser_field_frame(
    args: &Args,
    time_seconds: f64,
    frame_index: usize,
    audio: AudioFeature,
    frame: &mut [u8],
    stats: &mut FrameStats,
) -> String {
    let width = args.width;
    let height = args.height;
    let t = time_seconds as f32;
    let mut glow_pixels = 0_u64;
    let mut beam_accum = 0.0_f64;
    let beam_count = 24_usize;
    let threads = args.render_threads.min(height.max(1));
    let rows_per_chunk = height.div_ceil(threads);
    let row_stride = width * 3;

    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, chunk) in frame.chunks_mut(row_stride * rows_per_chunk).enumerate() {
            let y_start = chunk_index * rows_per_chunk;
            let rows = chunk.len() / row_stride;
            handles.push(scope.spawn(move || {
                let mut local_glow = 0_u64;
                let mut local_beam = 0.0_f64;
                for local_y in 0..rows {
                    let y = y_start + local_y;
                    let yf = y as f32 / height as f32;
                    for x in 0..width {
                        let xf = x as f32 / width as f32;
                        let sample = warp_laser_pixel(xf, yf, width, height, t, audio, beam_count);
                        if sample.glow {
                            local_glow += 1;
                        }
                        local_beam += sample.beam as f64;
                        write_pixel_local(chunk, width, x, local_y, sample.color);
                    }
                }
                (local_glow, local_beam)
            }));
        }
        for handle in handles {
            let (local_glow, local_beam) = handle.join().unwrap_or((0, 0.0));
            glow_pixels += local_glow;
            beam_accum += local_beam;
        }
    });

    stats.fog_samples += 1;
    stats.glow_pixels_sum += glow_pixels;

    format!(
        "{{\"frame_index\":{},\"time_seconds\":{:.6},\"scene\":\"warp_laser_field\",\"palette\":\"{}\",\"audio\":{{\"rms\":{:.6},\"bass\":{:.6},\"high\":{:.6},\"beat\":{:.6}}},\"render_law\":\"pure_black_background_no_fog_center_radial_laser_warp\",\"beam_count\":{},\"beam_mean\":{:.6},\"glow_pixels\":{},\"state_layers\":[\"pure_black_background\",\"center_origin_lasers\",\"radial_warp_starfield\",\"beat_pulse_core\",\"audio_reactive_beam_pressure\",\"no_fog\",\"no_city\",\"no_panels\"]}}",
        frame_index,
        time_seconds,
        json_escape(&args.palette),
        audio.rms,
        audio.bass,
        audio.high,
        audio.beat,
        beam_count,
        beam_accum / (width * height) as f64,
        glow_pixels
    )
}

#[derive(Clone, Copy)]
struct WarpPixel {
    color: Color,
    glow: bool,
    beam: f32,
}

fn warp_laser_pixel(
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    t: f32,
    audio: AudioFeature,
    beam_count: usize,
) -> WarpPixel {
    let aspect = width as f32 / height as f32;
    let cx = (x - 0.5) * aspect;
    let cy = y - 0.5;
    let radius = (cx * cx + cy * cy).sqrt();
    let angle = cy.atan2(cx);
    let mut color = Color::new(0.0, 0.0, 0.0);
    let mut glow = false;
    let mut beam_value = 0.0_f32;

    let core = warp_core(radius, audio);
    if core > 0.01 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(196.0, 118.0 + 72.0 * audio.high, 230.0 + 20.0 * audio.beat),
            core,
        );
    }

    let beam = warp_laser_beam_field(radius, angle, t, audio, beam_count);
    if beam.intensity > 0.01 {
        glow = true;
        beam_value = beam.intensity;
        blend_add(&mut color, beam.color, beam.intensity);
    }

    let stars = warp_starfield_streaks(cx, cy, radius, angle, t, audio);
    if stars.intensity > 0.01 {
        glow = true;
        blend_add(&mut color, stars.color, stars.intensity);
    }

    let tunnel = warp_tunnel_rings(radius, t, audio);
    if tunnel > 0.012 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(112.0 + 58.0 * audio.high, 72.0 + 50.0 * audio.rms, 180.0),
            tunnel,
        );
    }

    WarpPixel {
        color,
        glow,
        beam: beam_value,
    }
}

fn write_pixel_local(frame: &mut [u8], width: usize, x: usize, local_y: usize, color: Color) {
    let idx = (local_y * width + x) * 3;
    frame[idx] = color.b.clamp(0.0, 255.0) as u8;
    frame[idx + 1] = color.g.clamp(0.0, 255.0) as u8;
    frame[idx + 2] = color.r.clamp(0.0, 255.0) as u8;
}

#[derive(Clone, Copy)]
struct LightSample {
    intensity: f32,
    color: Color,
}

fn warp_core(radius: f32, audio: AudioFeature) -> f32 {
    let size = 0.026 + 0.018 * audio.bass + 0.010 * audio.beat;
    (-((radius / size.max(0.001)).powf(2.0))).exp() * (0.64 + 0.42 * audio.beat + 0.32 * audio.rms)
}

fn warp_laser_beam_field(
    radius: f32,
    angle: f32,
    t: f32,
    audio: AudioFeature,
    beam_count: usize,
) -> LightSample {
    if radius < 0.018 {
        return LightSample {
            intensity: 0.0,
            color: Color::default(),
        };
    }
    let mut intensity = 0.0_f32;
    let mut color = Color::default();
    for index in 0..beam_count {
        let fi = index as f32;
        let base = fi / beam_count as f32 * std::f32::consts::PI * 2.0;
        let wobble = 0.026 * (t * (0.58 + 0.013 * fi) + fi * 1.73).sin();
        let axis = base + wobble;
        let diff = angle_delta(angle, axis).abs();
        let width = 0.0034 + 0.0044 * audio.rms + 0.0018 * ((fi * 2.17 + t * 1.4).sin().abs());
        let core = 1.0 - smoothstep(width * 0.18, width, diff);
        let halo = 1.0 - smoothstep(width, width * 4.6, diff);
        let travel = (t * (0.36 + 0.42 * audio.bass + 0.25 * audio.beat) + fi * 0.079).fract();
        let pulse = 1.0 - smoothstep(0.0, 0.23, (radius - travel).abs());
        let reach = smoothstep(0.020, 0.16, radius) * (1.0 - smoothstep(1.05, 1.36, radius));
        let drive = (core * 1.0 + halo * 0.24) * (0.55 + 0.62 * pulse) * reach;
        let beam_strength = drive * (0.30 + 0.40 * audio.rms + 0.42 * audio.beat);
        if beam_strength > intensity {
            intensity = beam_strength;
            color = warp_laser_color(index, beam_count, audio);
        }
    }
    LightSample {
        intensity: intensity.clamp(0.0, 1.2),
        color,
    }
}

fn warp_laser_color(index: usize, beam_count: usize, audio: AudioFeature) -> Color {
    let band = index as f32 / beam_count.max(1) as f32;
    let blue = Color::new(255.0, 92.0, 34.0);
    let violet = Color::new(230.0, 58.0, 186.0);
    let gold = Color::new(24.0, 190.0, 255.0);
    let red = Color::new(22.0, 46.0, 255.0);
    let base = if band < 0.34 {
        lerp_color(blue, violet, band / 0.34)
    } else if band < 0.72 {
        lerp_color(violet, gold, (band - 0.34) / 0.38)
    } else {
        lerp_color(gold, red, (band - 0.72) / 0.28)
    };
    let pulse = 0.78 + 0.38 * audio.rms + 0.42 * audio.high + 0.24 * audio.beat;
    Color::new(base.b * pulse, base.g * pulse, base.r * pulse)
}

fn warp_starfield_streaks(
    cx: f32,
    cy: f32,
    radius: f32,
    angle: f32,
    t: f32,
    audio: AudioFeature,
) -> LightSample {
    let lanes = 72.0;
    let lane = ((angle + std::f32::consts::PI) / (std::f32::consts::PI * 2.0) * lanes).floor();
    let lane_phase = hash2(lane, 19.0);
    let lane_angle = lane / lanes * std::f32::consts::PI * 2.0 - std::f32::consts::PI;
    let diff = angle_delta(angle, lane_angle).abs();
    let width = 0.0022 + 0.0012 * audio.high;
    let line = 1.0 - smoothstep(width * 0.2, width, diff);
    let speed = 0.54 + 0.68 * audio.rms + 0.42 * audio.beat;
    let position = (radius * 1.18 - t * speed - lane_phase).fract();
    let head = 1.0 - smoothstep(0.0, 0.070 + 0.025 * audio.high, position);
    let tail = 1.0 - smoothstep(0.070, 0.36, position);
    let radial_gate = smoothstep(0.08, 0.24, radius) * (1.0 - smoothstep(1.02, 1.40, radius));
    let shimmer = 0.75 + 0.25 * (cx * 19.0 + cy * 17.0 + t * 2.1 + lane_phase * 6.0).sin();
    let intensity =
        line * (head * 0.86 + tail * 0.28) * radial_gate * shimmer * (0.32 + 0.76 * audio.high);
    LightSample {
        intensity: intensity.clamp(0.0, 1.0),
        color: Color::new(216.0, 176.0 + 52.0 * audio.high, 255.0),
    }
}

fn warp_tunnel_rings(radius: f32, t: f32, audio: AudioFeature) -> f32 {
    let phase = radius * (28.0 + 5.0 * audio.bass) - t * (1.15 + 0.85 * audio.rms);
    let ring = 1.0 - smoothstep(0.018, 0.090 + 0.020 * audio.beat, phase.sin().abs());
    let gate = smoothstep(0.08, 0.28, radius) * (1.0 - smoothstep(0.74, 1.22, radius));
    (ring * gate * (0.060 + 0.22 * audio.beat + 0.12 * audio.high)).clamp(0.0, 0.40)
}

fn render_memory_cathedral_frame(
    args: &Args,
    time_seconds: f64,
    frame_index: usize,
    audio: AudioFeature,
    frame: &mut [u8],
    stats: &mut FrameStats,
) -> String {
    let width = args.width;
    let height = args.height;
    let t = time_seconds as f32;
    let phase = (time_seconds / args.duration.max(0.000_001)).clamp(0.0, 1.0) as f32;
    let threads = args.render_threads.min(height.max(1));
    let rows_per_chunk = height.div_ceil(threads);
    let row_stride = width * 3;
    let mut glow_pixels = 0_u64;
    let mut veil_accum = 0.0_f64;
    let mut absence_accum = 0.0_f64;

    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, chunk) in frame.chunks_mut(row_stride * rows_per_chunk).enumerate() {
            let y_start = chunk_index * rows_per_chunk;
            let rows = chunk.len() / row_stride;
            handles.push(scope.spawn(move || {
                let mut local_glow = 0_u64;
                let mut local_veil = 0.0_f64;
                let mut local_absence = 0.0_f64;
                for local_y in 0..rows {
                    let y = y_start + local_y;
                    let yf = y as f32 / height as f32;
                    for x in 0..width {
                        let xf = x as f32 / width as f32;
                        let sample =
                            memory_cathedral_pixel(xf, yf, width, height, t, phase, audio);
                        if sample.glow {
                            local_glow += 1;
                        }
                        local_veil += sample.veil as f64;
                        local_absence += sample.absence as f64;
                        write_pixel_local(chunk, width, x, local_y, sample.color);
                    }
                }
                (local_glow, local_veil, local_absence)
            }));
        }
        for handle in handles {
            let (local_glow, local_veil, local_absence) = handle.join().unwrap_or((0, 0.0, 0.0));
            glow_pixels += local_glow;
            veil_accum += local_veil;
            absence_accum += local_absence;
        }
    });

    let pixel_count = (width * height).max(1) as f64;
    let veil_mean = veil_accum / pixel_count;
    let absence_mean = absence_accum / pixel_count;
    stats.fog_coverage_sum += veil_mean;
    stats.fog_samples += 1;
    stats.glow_pixels_sum += glow_pixels;

    format!(
        "{{\"frame_index\":{},\"time_seconds\":{:.6},\"scene\":\"memory_cathedral\",\"palette\":\"{}\",\"audio\":{{\"rms\":{:.6},\"bass\":{:.6},\"high\":{:.6},\"beat\":{:.6},\"vocal_presence\":{:.6}}},\"render_law\":\"state_fields_first_pixels_last_no_hard_edges\",\"phase\":{:.6},\"memory_veil_mean\":{:.6},\"absence_mean\":{:.6},\"glow_pixels\":{},\"state_layers\":[\"near_black_blue_memory_field\",\"soft_doorway_depth_windows\",\"central_human_absence_not_portrait\",\"left_right_voice_light_fields\",\"inward_memory_particles\",\"dream_snap_collapse_gate\",\"outro_heart_sink\",\"no_city\",\"no_fire\",\"no_hard_lasers\"]}}",
        frame_index,
        time_seconds,
        json_escape(&args.palette),
        audio.rms,
        audio.bass,
        audio.high,
        audio.beat,
        vocal_presence(audio),
        phase,
        veil_mean,
        absence_mean,
        glow_pixels
    )
}

#[derive(Clone, Copy)]
struct MemoryPixel {
    color: Color,
    glow: bool,
    veil: f32,
    absence: f32,
}

fn memory_cathedral_pixel(
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    t: f32,
    phase: f32,
    audio: AudioFeature,
) -> MemoryPixel {
    let aspect = width as f32 / height as f32;
    let cx = (x - 0.5) * aspect;
    let cy = y - 0.5;
    let radius = (cx * cx + cy * cy).sqrt();
    let angle = cy.atan2(cx);
    let vocal = vocal_presence(audio);
    let intro = 1.0 - smoothstep(0.00, 0.10, phase);
    let hook = phase_gate(phase, 0.27, 0.43) + phase_gate(phase, 0.78, 0.93);
    let dream_snap = phase_pulse(phase, 0.655, 0.026);
    let collapse = smoothstep(0.70, 0.91, phase);
    let outro = smoothstep(0.90, 1.0, phase);
    let breath = 0.5 + 0.5 * (t * (0.10 + 0.10 * audio.rms)).sin();

    let mut color = memory_base_field(x, y, radius, t, audio, intro, outro);
    let mut glow = false;

    let depth = memory_doorway_depth(cx, cy, t, phase, audio);
    if depth > 0.008 {
        glow = true;
        let door_color = lerp_color(
            Color::new(74.0, 56.0, 38.0),
            Color::new(116.0, 82.0 + 36.0 * vocal, 58.0 + 32.0 * audio.beat),
            (0.35 + 0.45 * hook + 0.20 * breath).clamp(0.0, 1.0),
        );
        blend_add(
            &mut color,
            door_color,
            depth * (0.24 + 0.18 * audio.rms + 0.22 * hook + 0.16 * collapse),
        );
    }

    let pair = memory_voice_pair_light(cx, cy, t, phase, audio);
    if pair > 0.010 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(106.0 + 54.0 * vocal, 96.0 + 38.0 * hook, 92.0 + 42.0 * audio.bass),
            pair,
        );
    }

    let waveform = memory_breath_wave(x, y, t, audio, hook);
    if waveform > 0.012 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(150.0 + 40.0 * audio.high, 118.0 + 20.0 * vocal, 126.0 + 32.0 * hook),
            waveform,
        );
    }

    let tunnel = memory_collapse_tunnel(radius, angle, t, phase, audio);
    if tunnel > 0.012 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(92.0 + 80.0 * audio.high, 70.0 + 48.0 * audio.rms, 104.0 + 70.0 * collapse),
            tunnel,
        );
    }

    let snap = memory_dream_snap(radius, angle, t, dream_snap, audio);
    if snap > 0.010 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(162.0 + 42.0 * audio.high, 138.0 + 42.0 * dream_snap, 162.0),
            snap,
        );
    }

    let particles = memory_inward_particles(cx, cy, radius, angle, t, audio, collapse);
    if particles > 0.014 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(154.0 + 38.0 * audio.high, 126.0 + 28.0 * audio.rms, 128.0 + 34.0 * hook),
            particles,
        );
    }

    let heart = memory_heart_sink(cx, cy, t, phase, audio);
    if heart > 0.010 {
        glow = true;
        blend_add(
            &mut color,
            Color::new(52.0 + 22.0 * audio.high, 78.0 + 60.0 * audio.bass, 142.0 + 58.0 * hook),
            heart,
        );
    }

    let absence = memory_human_absence(cx, cy, t, phase, audio);
    if absence > 0.0 {
        blend(&mut color, Color::new(0.5, 0.8, 1.2), absence * (0.58 + 0.18 * dream_snap));
        let rim = memory_absence_rim(cx, cy, t, phase, audio);
        if rim > 0.006 {
            glow = true;
            blend_add(
                &mut color,
                Color::new(92.0 + 40.0 * audio.high, 72.0 + 24.0 * vocal, 88.0 + 22.0 * hook),
                rim,
            );
        }
    }

    let veil = memory_veil_field(x, y, t, audio, collapse, outro);
    blend(
        &mut color,
        Color::new(32.0 + 18.0 * audio.high, 28.0 + 15.0 * vocal, 34.0 + 20.0 * hook),
        veil,
    );

    let blackout = dream_snap * (0.16 + 0.22 * (1.0 - audio.beat));
    blend(&mut color, Color::new(0.0, 0.0, 0.0), blackout);
    let vignette = smoothstep(0.26, 0.94, radius) * (0.52 + 0.24 * outro);
    blend(&mut color, Color::new(0.0, 0.0, 1.0), vignette);

    MemoryPixel {
        color,
        glow,
        veil,
        absence,
    }
}

fn phase_gate(phase: f32, start: f32, end: f32) -> f32 {
    smoothstep(start, start + 0.055, phase) * (1.0 - smoothstep(end - 0.055, end, phase))
}

fn phase_pulse(phase: f32, center: f32, width: f32) -> f32 {
    (-(((phase - center) / width.max(0.0001)).powf(2.0))).exp()
}

fn memory_base_field(
    x: f32,
    y: f32,
    radius: f32,
    t: f32,
    audio: AudioFeature,
    intro: f32,
    outro: f32,
) -> Color {
    let vertical = smoothstep(0.02, 1.0, y);
    let drift = value_noise(x * 2.4 + t * 0.010, y * 2.0 - t * 0.006);
    let pulse = 0.10 + 0.22 * audio.rms + 0.08 * audio.bass;
    let top = Color::new(
        8.0 + 8.0 * drift + 8.0 * intro,
        6.0 + 5.0 * pulse,
        11.0 + 6.0 * drift,
    );
    let bottom = Color::new(
        4.0 + 8.0 * audio.bass,
        4.0 + 5.0 * drift,
        7.0 + 8.0 * pulse,
    );
    let mut color = lerp_color(top, bottom, vertical);
    let center_breath = (-((radius / (0.64 + 0.10 * audio.bass)).powf(2.0))).exp();
    blend_add(
        &mut color,
        Color::new(12.0 + 18.0 * audio.high, 10.0 + 12.0 * audio.rms, 16.0 + 10.0 * intro),
        center_breath * (0.07 + 0.12 * audio.rms) * (1.0 - 0.55 * outro),
    );
    color
}

fn memory_doorway_depth(cx: f32, cy: f32, t: f32, phase: f32, audio: AudioFeature) -> f32 {
    let mut sum = 0.0_f32;
    let depth_drive = 0.32 + 0.20 * audio.bass + 0.16 * audio.rms;
    for index in 0..7 {
        let fi = index as f32;
        let z = fi / 6.0;
        let width = 0.15 + z * (0.58 + 0.16 * depth_drive);
        let height = 0.18 + z * (0.42 + 0.08 * audio.rms);
        let y_shift = -0.020 + 0.030 * (t * 0.035 + fi * 1.7).sin();
        let drift = 0.018 * (t * (0.025 + 0.004 * fi) + fi).sin();
        let d = (cx - drift).abs() / width.max(0.001);
        let e = (cy - y_shift).abs() / height.max(0.001);
        let border = (d.max(e) - 1.0).abs();
        let line_width = 0.010 + 0.006 * audio.rms + 0.004 * (fi * 0.7 + t * 0.08).sin().abs();
        let line = 1.0 - smoothstep(line_width * 0.30, line_width, border);
        let gate = (1.0 - smoothstep(1.0, 1.22, d.max(e))) * smoothstep(0.36, 0.88, d.max(e));
        let fade = (1.0 - z * 0.62) * (0.54 + 0.28 * (phase * std::f32::consts::PI).sin().abs());
        sum += line * gate * fade;
    }
    (sum * (0.20 + 0.26 * audio.rms + 0.10 * audio.beat)).clamp(0.0, 1.0)
}

fn memory_voice_pair_light(cx: f32, cy: f32, t: f32, phase: f32, audio: AudioFeature) -> f32 {
    let hook = phase_gate(phase, 0.27, 0.43) + phase_gate(phase, 0.78, 0.93);
    let separation = 0.30 - 0.11 * hook + 0.022 * (t * 0.07).sin();
    let vertical = cy + 0.005 * (t * 0.11).sin();
    let left = (-(((cx + separation) / 0.16).powf(2.0) + (vertical / 0.32).powf(2.0))).exp();
    let right = (-(((cx - separation) / 0.16).powf(2.0) + (vertical / 0.32).powf(2.0))).exp();
    let bridge = (-((cx / (0.25 + 0.07 * hook)).powf(2.0) + (vertical / 0.11).powf(2.0))).exp();
    ((left + right) * (0.055 + 0.23 * hook + 0.10 * audio.rms)
        + bridge * hook * (0.10 + 0.16 * audio.beat))
        .clamp(0.0, 0.92)
}

fn memory_breath_wave(x: f32, y: f32, t: f32, audio: AudioFeature, hook: f32) -> f32 {
    let wave = smoothed_waveform_at(audio, x);
    let center = 0.515 + wave * (0.028 + 0.030 * hook + 0.018 * vocal_presence(audio));
    let width = 0.025 + 0.018 * audio.rms + 0.010 * hook;
    let base = 1.0 - smoothstep(0.0, width, (y - center).abs());
    let echo_center = 0.515 - wave * (0.020 + 0.014 * audio.bass) + 0.020 * (x * 4.0 + t * 0.18).sin();
    let echo = 1.0 - smoothstep(0.0, width * 2.2, (y - echo_center).abs());
    (base * 0.22 + echo * 0.10) * (0.30 + 0.38 * audio.rms + 0.24 * hook)
}

fn memory_collapse_tunnel(
    radius: f32,
    angle: f32,
    t: f32,
    phase: f32,
    audio: AudioFeature,
) -> f32 {
    let collapse = smoothstep(0.70, 0.91, phase);
    let ring_phase = radius * (18.0 + 12.0 * collapse + 2.0 * audio.bass)
        - t * (0.20 + 0.38 * collapse + 0.10 * audio.rms);
    let ring = 1.0 - smoothstep(0.018, 0.20, ring_phase.sin().abs());
    let spokes = 1.0 - smoothstep(0.030, 0.17, (angle * 8.0 + t * 0.06).sin().abs());
    let gate = smoothstep(0.08, 0.24, radius) * (1.0 - smoothstep(0.72, 1.16, radius));
    (ring * gate * (0.08 + 0.34 * collapse + 0.20 * audio.beat)
        + spokes * gate * collapse * (0.035 + 0.10 * audio.high))
        .clamp(0.0, 0.84)
}

fn memory_dream_snap(
    radius: f32,
    angle: f32,
    t: f32,
    dream_snap: f32,
    audio: AudioFeature,
) -> f32 {
    let crack = 1.0 - smoothstep(0.008, 0.070, (angle * 11.0 + t * 0.30).sin().abs());
    let ring = 1.0 - smoothstep(0.0, 0.055, (radius - (0.18 + 0.10 * dream_snap)).abs());
    let shock = 1.0 - smoothstep(0.22, 0.84, radius);
    (dream_snap * (ring * 0.58 + crack * shock * 0.22) * (0.32 + 0.32 * audio.high + 0.24 * audio.beat))
        .clamp(0.0, 1.0)
}

fn memory_inward_particles(
    cx: f32,
    cy: f32,
    radius: f32,
    angle: f32,
    t: f32,
    audio: AudioFeature,
    collapse: f32,
) -> f32 {
    let lanes = 92.0;
    let lane = ((angle + std::f32::consts::PI) / (std::f32::consts::PI * 2.0) * lanes).floor();
    let lane_phase = hash2(lane, 77.0);
    let lane_angle = lane / lanes * std::f32::consts::PI * 2.0 - std::f32::consts::PI;
    let diff = angle_delta(angle, lane_angle).abs();
    let line = 1.0 - smoothstep(0.004, 0.018 + 0.006 * audio.high, diff);
    let speed = 0.10 + 0.26 * audio.rms + 0.24 * collapse + 0.08 * audio.beat;
    let position = (radius * (1.4 + 0.20 * collapse) + t * speed + lane_phase).fract();
    let head = 1.0 - smoothstep(0.0, 0.050 + 0.030 * audio.high, position);
    let tail = 1.0 - smoothstep(0.05, 0.24, position);
    let breakup = value_noise(cx * 9.0 + t * 0.03 + lane_phase, cy * 9.0 - t * 0.02);
    let gate = smoothstep(0.10, 0.30, radius) * (1.0 - smoothstep(0.82, 1.30, radius));
    (line * (head * 0.84 + tail * 0.20) * gate * breakup * (0.16 + 0.42 * audio.high + 0.22 * collapse))
        .clamp(0.0, 0.78)
}

fn memory_heart_sink(cx: f32, cy: f32, t: f32, phase: f32, audio: AudioFeature) -> f32 {
    let outro = smoothstep(0.88, 1.0, phase);
    let y = cy + 0.02 - 0.06 * outro + 0.006 * (t * 0.09).sin();
    let x = cx + 0.004 * (t * 0.07).sin();
    let lobe_l = (-((((x + 0.035) / 0.050).powf(2.0)) + (((y + 0.012) / 0.038).powf(2.0)))).exp();
    let lobe_r = (-((((x - 0.035) / 0.050).powf(2.0)) + (((y + 0.012) / 0.038).powf(2.0)))).exp();
    let lower = (-(((x / 0.075).powf(2.0)) + (((y - 0.040) / 0.078).powf(2.0)))).exp();
    (lobe_l.max(lobe_r).max(lower) * (0.05 + 0.22 * outro + 0.16 * audio.bass + 0.10 * audio.rms))
        .clamp(0.0, 0.78)
}

fn memory_human_absence(cx: f32, cy: f32, t: f32, phase: f32, audio: AudioFeature) -> f32 {
    let drift = 0.010 * (t * 0.055).sin();
    let head = (-((((cx - drift) / 0.060).powf(2.0)) + (((cy + 0.170) / 0.085).powf(2.0)))).exp();
    let torso = (-((((cx - drift) / 0.112).powf(2.0)) + (((cy + 0.010) / 0.245).powf(2.0)))).exp();
    let vanish = smoothstep(0.88, 1.0, phase);
    let snap_void = phase_pulse(phase, 0.655, 0.038);
    ((head.max(torso * 0.92)) * (0.30 + 0.18 * vocal_presence(audio) + 0.20 * snap_void) * (1.0 - 0.62 * vanish))
        .clamp(0.0, 0.78)
}

fn memory_absence_rim(cx: f32, cy: f32, t: f32, phase: f32, audio: AudioFeature) -> f32 {
    let a = memory_human_absence(cx, cy, t, phase, audio);
    let b = memory_human_absence(cx + 0.010, cy, t, phase, audio)
        .min(memory_human_absence(cx - 0.010, cy, t, phase, audio))
        .min(memory_human_absence(cx, cy + 0.010, t, phase, audio))
        .min(memory_human_absence(cx, cy - 0.010, t, phase, audio));
    ((a - b).max(0.0) * (0.65 + 0.42 * audio.high + 0.22 * audio.beat)).clamp(0.0, 0.55)
}

fn memory_veil_field(
    x: f32,
    y: f32,
    t: f32,
    audio: AudioFeature,
    collapse: f32,
    outro: f32,
) -> f32 {
    let n1 = value_noise(x * 4.0 + t * 0.018, y * 3.5 - t * 0.012);
    let n2 = value_noise(x * 8.5 - t * 0.026, y * 5.5 + t * 0.019);
    let vertical = smoothstep(0.10, 0.82, y) * (1.0 - smoothstep(0.94, 1.0, y));
    let inward = 1.0 - ((x - 0.5).abs() * 1.40).clamp(0.0, 1.0);
    ((n1 * 0.16 + n2 * 0.10 + inward * 0.08)
        * vertical
        * (0.16 + 0.22 * audio.rms + 0.12 * collapse)
        * (1.0 - 0.48 * outro))
        .clamp(0.0, 0.36)
}

fn render_abstract_symphony_frame(
    args: &Args,
    time_seconds: f64,
    frame_index: usize,
    audio: AudioFeature,
    frame: &mut [u8],
    stats: &mut FrameStats,
) -> String {
    let width = args.width;
    let height = args.height;
    let t = time_seconds as f32;
    let aspect = width as f32 / height as f32;
    let energy = (0.22 + 0.96 * audio.rms + 0.28 * audio.beat).clamp(0.0, 1.35);
    let bass = audio.bass.clamp(0.0, 1.0);
    let high = audio.high.clamp(0.0, 1.0);
    let beat = audio.beat.clamp(0.0, 1.0);
    let vocal = vocal_presence(audio);
    let mut glow_pixels = 0_u64;
    let mut fog_accum = 0.0_f64;

    for y in 0..height {
        let yf = y as f32 / height as f32;
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let cx = (xf - 0.5) * aspect;
            let cy = yf - 0.5;
            let radius = (cx * cx + cy * cy).sqrt();
            let angle = cy.atan2(cx);

            let mut color = abstract_symphony_base(xf, yf, t, energy);
            let edge_fog_mask = abstract_edge_fog_mask(xf, yf);
            let field = abstract_symphony_fog_field(xf, yf, t, audio) * edge_fog_mask;
            fog_accum += field as f64;
            blend(
                &mut color,
                Color::new(48.0 + 62.0 * high, 28.0 + 36.0 * energy, 58.0 + 58.0 * beat),
                (field * (0.30 + 0.34 * energy)).clamp(0.0, 0.78),
            );

            let aurora = abstract_aurora_veil(xf, yf, t, audio);
            if aurora > 0.015 {
                glow_pixels += 1;
                let blue = Color::new(190.0, 82.0, 42.0);
                let purple = Color::new(156.0, 54.0, 184.0);
                let gold = Color::new(34.0, 150.0, 224.0);
                let band = 0.5 + 0.5 * (angle * 2.2 + t * 0.13).sin();
                let veil_color = if band < 0.58 {
                    lerp_color(blue, purple, band / 0.58)
                } else {
                    lerp_color(purple, gold, (band - 0.58) / 0.42)
                };
                blend_add(
                    &mut color,
                    veil_color,
                    aurora * (0.42 + 0.32 * audio.rms + 0.24 * beat),
                );
            }

            let beams = abstract_soft_club_beams(cx, cy, t, audio);
            if beams > 0.012 {
                glow_pixels += 1;
                let beam_color = lerp_color(
                    Color::new(184.0, 70.0, 36.0),
                    Color::new(26.0, 132.0, 218.0),
                    (0.22 + 0.58 * bass + 0.20 * beat).clamp(0.0, 1.0),
                );
                blend_add(
                    &mut color,
                    beam_color,
                    beams * (0.40 + 0.26 * audio.rms + 0.22 * beat),
                );
            }

            let ribbon = abstract_wave_ribbon(xf, yf, t, audio);
            if ribbon > 0.015 {
                glow_pixels += 1;
                blend_add(
                    &mut color,
                    Color::new(
                        178.0 + 42.0 * high,
                        78.0 + 68.0 * vocal,
                        184.0 + 52.0 * beat,
                    ),
                    ribbon * (0.42 + 0.38 * energy),
                );
            }

            let soundfield = abstract_soundfield_rings(radius, angle, t, audio);
            if soundfield > 0.015 {
                glow_pixels += 1;
                blend_add(
                    &mut color,
                    Color::new(130.0 + 72.0 * high, 64.0 + 62.0 * bass, 152.0 + 72.0 * beat),
                    soundfield * (0.30 + 0.32 * audio.rms),
                );
            }

            let low_glow = abstract_low_frequency_glow(cx, cy, t, bass, beat);
            if low_glow > 0.015 {
                glow_pixels += 1;
                blend_add(
                    &mut color,
                    Color::new(18.0 + 28.0 * high, 82.0 + 82.0 * bass, 184.0 + 58.0 * beat),
                    low_glow,
                );
            }

            let reflection = abstract_reflection_field(xf, yf, t, audio);
            if reflection > 0.01 {
                blend_add(
                    &mut color,
                    Color::new(
                        108.0 + 56.0 * high,
                        54.0 + 42.0 * energy,
                        116.0 + 86.0 * bass,
                    ),
                    reflection,
                );
            }

            let sparkle = abstract_soft_sparkle(xf, yf, t, high, beat);
            if sparkle > 0.02 {
                glow_pixels += 1;
                blend_add(
                    &mut color,
                    Color::new(172.0, 128.0 + 52.0 * beat, 210.0 + 40.0 * high),
                    sparkle,
                );
            }

            let vignette = smoothstep(0.34, 0.98, radius);
            blend(&mut color, Color::new(1.0, 1.0, 4.0), vignette * 0.64);
            write_pixel(frame, width, x, y, color);
        }
    }

    stats.fog_coverage_sum += fog_accum / (width * height) as f64;
    stats.fog_samples += 1;
    stats.glow_pixels_sum += glow_pixels;

    format!(
        "{{\"frame_index\":{},\"time_seconds\":{:.6},\"scene\":\"abstract_symphony\",\"palette\":\"{}\",\"audio\":{{\"rms\":{:.6},\"bass\":{:.6},\"high\":{:.6},\"beat\":{:.6},\"vocal_presence\":{:.6}}},\"render_law\":\"no_hard_edges_state_fields_only\",\"fog_mean\":{:.6},\"glow_pixels\":{},\"state_layers\":[\"soft_volumetric_fog_field\",\"audio_waveform_ribbon\",\"electric_glow_pressure\",\"soundfield_rings\",\"bokeh_sparks\",\"wet_reflection_pressure\",\"no_grid_presentation\",\"no_geometry_bars\"]}}",
        frame_index,
        time_seconds,
        json_escape(&args.palette),
        audio.rms,
        audio.bass,
        audio.high,
        audio.beat,
        vocal,
        fog_accum / (width * height) as f64,
        glow_pixels
    )
}

fn abstract_symphony_base(x: f32, y: f32, t: f32, energy: f32) -> Color {
    let vertical = smoothstep(0.02, 1.0, y);
    let drift = 0.5 + 0.5 * (x * 2.6 + y * 1.8 + t * 0.035).sin();
    let upper = Color::new(8.0 + 13.0 * drift, 5.0 + 6.0 * energy, 12.0 + 10.0 * drift);
    let lower = Color::new(4.0 + 10.0 * energy, 4.0 + 6.0 * drift, 8.0 + 17.0 * energy);
    lerp_color(upper, lower, vertical)
}

fn abstract_symphony_fog_field(x: f32, y: f32, t: f32, audio: AudioFeature) -> f32 {
    let n1 = value_noise(x * 4.6 + t * 0.035, y * 3.2 - t * 0.018);
    let n2 = value_noise(x * 9.0 - t * 0.050, y * 6.5 + t * 0.030);
    let band = 0.5 + 0.5 * (x * 8.0 - y * 5.2 + t * (0.18 + 0.12 * audio.bass)).sin();
    let height_gate = smoothstep(0.04, 0.62, y) * (1.0 - smoothstep(0.96, 1.0, y));
    ((0.22 * n1 + 0.18 * n2 + 0.34 * band) * height_gate * (0.74 + 0.46 * audio.rms))
        .clamp(0.0, 1.0)
}

fn abstract_edge_fog_mask(x: f32, y: f32) -> f32 {
    let left = 1.0 - smoothstep(0.02, 0.35, x);
    let right = smoothstep(0.65, 0.98, x);
    let top = (1.0 - smoothstep(0.02, 0.30, y)) * 0.55;
    let bottom = smoothstep(0.72, 0.99, y) * 0.72;
    let corner_push = ((x - 0.5).abs() * 1.35 + (y - 0.50).abs() * 0.82).clamp(0.0, 1.0);
    (left.max(right).max(top).max(bottom) * 0.86 + corner_push * 0.22).clamp(0.0, 1.0)
}

fn abstract_aurora_veil(x: f32, y: f32, t: f32, audio: AudioFeature) -> f32 {
    let phase = y * 7.2
        + (x * 5.0 + t * (0.10 + 0.10 * audio.high)).sin() * 0.95
        + value_noise(x * 3.0 + t * 0.02, y * 2.0) * 1.25;
    let fold = 1.0 - smoothstep(0.12, 0.58, phase.sin().abs());
    let lane = smoothstep(0.08, 0.30, y) * (1.0 - smoothstep(0.74, 0.97, y));
    (fold * lane * (0.20 + 0.62 * audio.high + 0.34 * audio.beat)).clamp(0.0, 1.0)
}

fn abstract_soft_club_beams(cx: f32, cy: f32, t: f32, audio: AudioFeature) -> f32 {
    let origin_y = 0.43 + 0.018 * (t * 0.09).sin();
    let dx = cx;
    let dy = cy - origin_y;
    let distance = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);
    let axes = [
        -2.40 + 0.16 * (t * 0.11).sin(),
        -2.02 + 0.20 * (t * 0.13 + 1.7).sin(),
        -1.55 + 0.18 * (t * 0.10 + 2.9).sin(),
        -1.10 + 0.21 * (t * 0.12 + 4.2).sin(),
        -0.73 + 0.15 * (t * 0.14 + 5.1).sin(),
    ];
    let mut beam_sum = 0.0_f32;
    for (index, axis) in axes.iter().enumerate() {
        let phase = index as f32 * 1.31;
        let diff = angle_delta(angle, *axis).abs();
        let width = 0.040 + 0.034 * audio.rms + 0.018 * (phase + t * 0.22).sin().abs();
        let core = 1.0 - smoothstep(width * 0.22, width, diff);
        let halo = 1.0 - smoothstep(width, width * 3.2, diff);
        let reach = smoothstep(0.02, 0.36, distance) * (1.0 - smoothstep(0.92, 1.45, distance));
        let breakup = 0.62
            + 0.38
                * value_noise(
                    cx * 4.2 + t * (0.05 + 0.03 * audio.high) + phase,
                    cy * 7.0 - t * 0.04,
                );
        beam_sum += (core * 0.92 + halo * 0.28) * reach * breakup;
    }
    (beam_sum * (0.18 + 0.46 * audio.bass + 0.20 * audio.beat)).clamp(0.0, 1.0)
}

fn abstract_wave_ribbon(x: f32, y: f32, t: f32, audio: AudioFeature) -> f32 {
    let wave = smoothed_waveform_at(audio, (x + 0.018 * (t * 0.22).sin()).clamp(0.0, 1.0));
    let memory = 0.5 + 0.5 * (x * 7.0 + t * 0.7).sin();
    let center = 0.51 + wave * (0.040 + 0.030 * vocal_presence(audio)) + (memory - 0.5) * 0.022;
    let width = 0.060 + 0.042 * audio.rms + 0.030 * audio.beat;
    let main = 1.0 - smoothstep(0.0, width, (y - center).abs());
    let upper = 1.0
        - smoothstep(
            0.0,
            width * 2.4,
            (y - center - 0.11 - 0.020 * audio.bass).abs(),
        );
    let lower = 1.0
        - smoothstep(
            0.0,
            width * 2.7,
            (y - center + 0.13 + 0.016 * audio.high).abs(),
        );
    (main * 0.42 + upper * 0.18 + lower * 0.16).clamp(0.0, 1.0)
}

fn abstract_soundfield_rings(radius: f32, angle: f32, t: f32, audio: AudioFeature) -> f32 {
    let stereo = 0.5 + 0.5 * (angle * 2.0 + t * 0.32).sin();
    let pulse = audio.rms * 0.55 + audio.beat * 0.50 + stereo * 0.12;
    let ring_phase = radius * (17.0 + 1.8 * audio.bass) - t * (0.42 + 0.34 * audio.high);
    let ring = 1.0 - smoothstep(0.02, 0.46, ring_phase.sin().abs());
    let center_gate = 1.0 - smoothstep(0.08, 0.74, radius);
    (ring * center_gate * (0.10 + 0.48 * pulse)).clamp(0.0, 1.0)
}

fn abstract_low_frequency_glow(cx: f32, cy: f32, t: f32, bass: f32, beat: f32) -> f32 {
    let y = cy + 0.34 + 0.026 * (t * 0.18).sin();
    let x1 = cx - 0.22 * (t * 0.11).sin();
    let x2 = cx + 0.32 * (t * 0.09 + 1.4).sin();
    let left = (-((x1 / 0.42).powf(2.0) + (y / (0.17 + 0.08 * bass)).powf(2.0))).exp();
    let right = (-((x2 / 0.36).powf(2.0) + (y / (0.14 + 0.06 * beat)).powf(2.0))).exp();
    ((left + right * 0.74) * (0.22 + 0.70 * bass + 0.28 * beat)).clamp(0.0, 1.0)
}

fn abstract_reflection_field(x: f32, y: f32, t: f32, audio: AudioFeature) -> f32 {
    let floor_gate = smoothstep(0.52, 1.0, y);
    let ripple = 0.5 + 0.5 * (x * 22.0 + y * 14.0 + t * (0.85 + 0.45 * audio.bass)).sin();
    let smear = value_noise(x * 11.0 - t * 0.08, y * 21.0 + t * 0.05);
    (floor_gate * (0.18 * ripple + 0.16 * smear) * (0.36 + 0.72 * audio.bass + 0.20 * audio.beat))
        .clamp(0.0, 0.70)
}

fn abstract_soft_sparkle(x: f32, y: f32, t: f32, high: f32, beat: f32) -> f32 {
    let drift = value_noise(x * 18.0 + t * 0.18, y * 13.0 - t * 0.14);
    let cloud = value_noise(x * 7.0 - t * 0.04, y * 6.0 + t * 0.03);
    let gate = smoothstep(
        0.58 - 0.14 * high - 0.06 * beat,
        1.0,
        drift * 0.52 + cloud * 0.48,
    );
    let height = smoothstep(0.06, 0.38, y) * (1.0 - smoothstep(0.88, 0.98, y));
    (gate * height * (0.04 + 0.24 * high + 0.12 * beat)).clamp(0.0, 0.38)
}

fn waveform_at(audio: AudioFeature, x: f32) -> f32 {
    let pos = x.clamp(0.0, 1.0) * (WAVEFORM_BINS - 1) as f32;
    let left = pos.floor() as usize;
    let right = (left + 1).min(WAVEFORM_BINS - 1);
    let mix = pos.fract();
    audio.waveform[left] * (1.0 - mix) + audio.waveform[right] * mix
}

fn smoothed_waveform_at(audio: AudioFeature, x: f32) -> f32 {
    let mut sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for offset in -3..=3 {
        let distance = (offset as f32).abs();
        let weight = 1.0 / (1.0 + distance * distance);
        let sample_x = (x + offset as f32 / (WAVEFORM_BINS - 1) as f32).clamp(0.0, 1.0);
        sum += waveform_at(audio, sample_x) * weight;
        weight_sum += weight;
    }
    sum / weight_sum.max(0.0001)
}

fn angle_delta(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::PI * 2.0;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::PI * 2.0;
    }
    d
}

fn background_color(x: f32, y: f32, horizon: f32, t: f64) -> Color {
    let sky = smoothstep(0.0, horizon + 0.22, y);
    let storm = 0.5 + 0.5 * ((x * 7.0 + y * 5.0 + t as f32 * 0.08).sin());
    let base = Color::new(
        13.0 + 24.0 * (1.0 - y) + 10.0 * storm,
        17.0 + 20.0 * (1.0 - y),
        27.0 + 22.0 * (1.0 - y),
    );
    let ground = Color::new(9.0 + 12.0 * x, 12.0 + 7.0 * x, 17.0 + 5.0 * x);
    lerp_color(base, ground, sky)
}

fn depth_at(x: f32, y: f32, t: f64) -> f32 {
    let tunnel = ((x - 0.5).abs() * 1.7 + (y - 0.48).max(0.0) * 1.1).min(1.0);
    let drift = 0.08 * (t as f32 * 0.21 + x * 3.0).sin();
    (0.15 + 0.85 * tunnel + drift).clamp(0.0, 1.0)
}

fn apply_ground_reflection(color: &mut Color, x: f32, y: f32, t: f64, pulse: f32) {
    if y < 0.48 {
        return;
    }
    let lane = (1.0 - (x - 0.5).abs() * 1.8).max(0.0);
    let ripple = 0.5 + 0.5 * (x * 42.0 + y * 18.0 + t as f32 * 2.8).sin();
    let alpha = lane * ripple * (y - 0.48) * 0.42;
    blend_add(
        color,
        Color::new(10.0, 42.0 + 24.0 * pulse, 86.0 + 42.0 * pulse),
        alpha,
    );
}

fn portal_glow(x: f32, y: f32, t: f64) -> f32 {
    let cx = 0.5 + 0.025 * (t * 0.2).sin() as f32;
    let cy = 0.43;
    let dx = (x - cx) / 0.17;
    let dy = (y - cy) / 0.25;
    let d = dx * dx + dy * dy;
    (1.0 - smoothstep(0.05, 1.0, d)).max(0.0)
}

fn build_pillars(width: usize, height: usize, t: f64) -> Vec<Rect> {
    let drift = (t * 0.13).sin() as i32 * 8;
    vec![
        Rect {
            x0: (width as f32 * 0.07) as i32 + drift,
            y0: 0,
            x1: (width as f32 * 0.15) as i32 + drift,
            y1: height as i32,
        },
        Rect {
            x0: (width as f32 * 0.78) as i32 - drift,
            y0: 0,
            x1: (width as f32 * 0.88) as i32 - drift,
            y1: height as i32,
        },
        Rect {
            x0: (width as f32 * 0.38) as i32,
            y0: 0,
            x1: (width as f32 * 0.43) as i32,
            y1: (height as f32 * 0.74) as i32,
        },
    ]
}

fn build_silhouettes(width: usize, height: usize, t: f64) -> Vec<Rect> {
    let walk = ((t * 0.10).sin() * width as f64 * 0.07) as i32;
    vec![
        Rect {
            x0: (width as f32 * 0.53) as i32 + walk,
            y0: (height as f32 * 0.42) as i32,
            x1: (width as f32 * 0.60) as i32 + walk,
            y1: (height as f32 * 0.82) as i32,
        },
        Rect {
            x0: (width as f32 * 0.30) as i32 - walk / 2,
            y0: (height as f32 * 0.50) as i32,
            x1: (width as f32 * 0.35) as i32 - walk / 2,
            y1: (height as f32 * 0.78) as i32,
        },
    ]
}

fn silhouette_alpha(x: i32, y: i32, silhouettes: &[Rect]) -> f32 {
    for rect in silhouettes {
        let cx = (rect.x0 + rect.x1) as f32 * 0.5;
        let w = (rect.x1 - rect.x0).max(1) as f32 * 0.5;
        let h = (rect.y1 - rect.y0).max(1) as f32;
        let nx = (x as f32 - cx) / w;
        let ny = (y as f32 - rect.y0 as f32) / h;
        let body = 1.0 - (nx * nx + (ny - 0.55).powf(2.0) * 1.8);
        let head = 1.0 - (nx * 1.7).powf(2.0) - ((ny - 0.11) * 6.0).powf(2.0);
        let a = body.max(head).clamp(0.0, 1.0);
        if a > 0.0 {
            return a;
        }
    }
    0.0
}

fn pillar_alpha(x: i32, y: i32, pillars: &[Rect]) -> f32 {
    for rect in pillars {
        if rect.contains(x, y) {
            return 0.92;
        }
    }
    0.0
}

fn pillar_rim(x: i32, y: i32, pillars: &[Rect]) -> f32 {
    for rect in pillars {
        if y < rect.y0 || y > rect.y1 {
            continue;
        }
        let d = (x - rect.x0).abs().min((x - rect.x1).abs());
        if d <= 3 {
            return 1.0 - d as f32 / 3.0;
        }
    }
    0.0
}

fn fog_density(x: f32, y: f32, depth: f32, t: f64, frame_index: usize) -> f32 {
    let band1 = 0.5 + 0.5 * (x * 10.0 + y * 6.0 + t as f32 * 0.42).sin();
    let band2 = 0.5 + 0.5 * (x * 27.0 - y * 12.0 - t as f32 * 0.23).sin();
    let hash = value_noise(
        x * 9.0 + t as f32 * 0.06,
        y * 6.0 + frame_index as f32 * 0.001,
    );
    let vertical = smoothstep(0.20, 0.84, y) * (1.0 - smoothstep(0.90, 1.0, y));
    (0.08 + 0.30 * band1 + 0.18 * band2 + 0.22 * hash) * vertical * (0.35 + 0.75 * depth)
}

fn apply_fog(color: &mut Color, fog: f32, veil: f32, y: f32) {
    let alpha = (fog * (0.38 + 0.20 * veil)).clamp(0.0, 0.74);
    let fog_color = Color::new(
        36.0 + 20.0 * (1.0 - y),
        42.0 + 18.0 * (1.0 - y),
        52.0 + 16.0 * (1.0 - y),
    );
    blend(color, fog_color, alpha);
}

fn ember_field(x: f32, y: f32, t: f64, high: f32, beat: f32) -> f32 {
    let lanes = (x * 34.0 + t as f32 * 1.4).sin() * (y * 17.0 - t as f32 * 0.8).cos();
    let spots = value_noise(x * 80.0 + t as f32 * 3.2, y * 52.0 - t as f32 * 1.7);
    let threshold = 0.86 - 0.10 * high - 0.06 * beat;
    if lanes > threshold && spots > 0.72 && y > 0.16 && y < 0.86 {
        ((lanes - threshold) * 3.8 * spots * (0.55 + 0.65 * high + 0.35 * beat)).min(0.9)
    } else {
        0.0
    }
}

fn write_pixel(frame: &mut [u8], width: usize, x: usize, y: usize, color: Color) {
    let idx = (y * width + x) * 3;
    frame[idx] = color.b.clamp(0.0, 255.0) as u8;
    frame[idx + 1] = color.g.clamp(0.0, 255.0) as u8;
    frame[idx + 2] = color.r.clamp(0.0, 255.0) as u8;
}

fn blend(color: &mut Color, other: Color, alpha: f32) {
    let a = alpha.clamp(0.0, 1.0);
    color.b = color.b * (1.0 - a) + other.b * a;
    color.g = color.g * (1.0 - a) + other.g * a;
    color.r = color.r * (1.0 - a) + other.r * a;
}

fn blend_add(color: &mut Color, other: Color, alpha: f32) {
    let a = alpha.max(0.0);
    color.b = (color.b + other.b * a).min(255.0);
    color.g = (color.g + other.g * a).min(255.0);
    color.r = (color.r + other.r * a).min(255.0);
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let x = t.clamp(0.0, 1.0);
    Color::new(
        a.b * (1.0 - x) + b.b * x,
        a.g * (1.0 - x) + b.g * x,
        a.r * (1.0 - x) + b.r * x,
    )
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let x = ((value - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn value_noise(x: f32, y: f32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let a = hash2(xi, yi);
    let b = hash2(xi + 1.0, yi);
    let c = hash2(xi, yi + 1.0);
    let d = hash2(xi + 1.0, yi + 1.0);
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let x1 = a * (1.0 - u) + b * u;
    let x2 = c * (1.0 - u) + d * u;
    x1 * (1.0 - v) + x2 * v
}

fn hash2(x: f32, y: f32) -> f32 {
    let n = (x * 127.1 + y * 311.7).sin() * 43758.547;
    n.fract().abs()
}

fn process_memory_snapshot() -> (u64, u64) {
    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        PageFaultCount: 0,
        PeakWorkingSetSize: 0,
        WorkingSetSize: 0,
        QuotaPeakPagedPoolUsage: 0,
        QuotaPagedPoolUsage: 0,
        QuotaPeakNonPagedPoolUsage: 0,
        QuotaNonPagedPoolUsage: 0,
        PagefileUsage: 0,
        PeakPagefileUsage: 0,
    };
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    };
    if ok == 0 {
        (0, 0)
    } else {
        (
            counters.WorkingSetSize as u64,
            counters.PeakWorkingSetSize as u64,
        )
    }
}

fn manifest_claim(scene_mode: &str) -> &'static str {
    match scene_mode {
        "memory_cathedral" | "fade_away_memory_cathedral" => {
            "deterministic synthetic state-media render for ambient memory, dream collapse, absence, voice light, and inward fade"
        }
        "warp_laser_field" | "laser_warp" => {
            "deterministic synthetic state-media render for center-origin laser warp fields and audio-reactive beam pressure"
        }
        "lyric_city" => {
            "deterministic synthetic state-media render for lyric-guided city silhouettes, audio windows, fog, glow, and emotional scene pressure"
        }
        "spectrum_backdrop" => {
            "deterministic source-poster intensity animation for existing electric, analyzer, waveform, and soundfield regions"
        }
        "abstract_symphony" | "symphony" => {
            "deterministic synthetic state-media render for abstract audio fields, soft beams, and soundfield pressure"
        }
        _ => "deterministic synthetic state-media demo for occlusion, mist, fog, glow, and depth",
    }
}

fn scene_motifs_json(scene_mode: &str) -> &'static str {
    match scene_mode {
        "memory_cathedral" | "fade_away_memory_cathedral" => {
            "[\"near-black blue memory field\", \"soft doorway depth windows\", \"central human absence\", \"paired voice light fields\", \"inward memory particles\", \"dream snap collapse\", \"outro heart sink\"]"
        }
        "warp_laser_field" | "laser_warp" => {
            "[\"pure black background\", \"center-origin lasers\", \"radial warp starfield\", \"beat pulse core\", \"audio-reactive beam pressure\"]"
        }
        "lyric_city" => {
            "[\"wide night sky\", \"black skyline silhouettes\", \"bottom-up audio windows\", \"memory fog\", \"wet reflections\", \"father-child emotional arc\"]"
        }
        "spectrum_backdrop" => {
            "[\"full source poster\", \"existing electric intensity\", \"audio waveform panel\", \"spectrum analyzer panel\", \"headphone soundfield radar\", \"prototype status mark\"]"
        }
        "abstract_symphony" | "symphony" => {
            "[\"soft volumetric field\", \"audio waveform ribbon\", \"electric glow pressure\", \"soundfield rings\", \"wet reflection pressure\"]"
        }
        _ => {
            "[\"foreground occlusion pillars\", \"moving silhouette occlusion\", \"depth fog veils\", \"portal glow\", \"wet reflection\", \"ember pressure\"]"
        }
    }
}

fn write_manifest(
    args: &Args,
    video_path: &PathBuf,
    visual_path: &PathBuf,
    state_path: &PathBuf,
    manifest_path: &PathBuf,
    frame_count: usize,
    wall_seconds: f64,
    memory_start: (u64, u64),
    memory_end: (u64, u64),
    stats: &FrameStats,
) -> Result<(), String> {
    let avg_fog = if stats.fog_samples == 0 {
        0.0
    } else {
        stats.fog_coverage_sum / stats.fog_samples as f64
    };
    let avg_occluded = if frame_count == 0 {
        0.0
    } else {
        stats.occluded_pixels_sum as f64 / frame_count as f64
    };
    let avg_glow = if frame_count == 0 {
        0.0
    } else {
        stats.glow_pixels_sum as f64 / frame_count as f64
    };
    let mut file = BufWriter::new(
        File::create(manifest_path).map_err(|e| format!("manifest open failed: {e}"))?,
    );
    writeln!(
        file,
        concat!(
            "{{\n",
            "  \"schema\": \"truevision_weird_occlusion_rs.v1\",\n",
            "  \"run_id\": \"{}\",\n",
            "  \"created_at_unix\": {},\n",
            "  \"claim\": \"{}\",\n",
            "  \"boundary\": {{\n",
            "    \"synthetic_state_media\": true,\n",
            "    \"no_external_visual_assets\": true,\n",
            "    \"no_python_render_loop\": true,\n",
            "    \"rust_hot_path\": true\n",
            "  }},\n",
            "  \"render\": {{\n",
            "    \"video_path\": \"{}\",\n",
            "    \"visual_only_path\": \"{}\",\n",
            "    \"frame_state_jsonl\": \"{}\",\n",
            "    \"width\": {},\n",
            "    \"height\": {},\n",
            "    \"fps\": {},\n",
            "    \"duration_seconds\": {:.6},\n",
            "    \"frame_count\": {},\n",
            "    \"encoder\": \"ffmpeg {}\",\n",
            "    \"crf\": {},\n",
            "    \"bitrate\": \"{}\",\n",
            "    \"render_threads\": {},\n",
            "    \"state_log_every\": {}\n",
            "  }},\n",
            "  \"audio\": {{\n",
            "    \"path\": {},\n",
            "    \"sample_rate_used\": {},\n",
            "    \"ffmpeg_observer\": {},\n",
            "    \"muxed_into_output\": {}\n",
            "  }},\n",
            "  \"scene_state\": {{\n",
            "    \"scene_mode\": \"{}\",\n",
            "    \"palette\": \"{}\",\n",
            "    \"motifs\": {},\n",
            "    \"average_fog_coverage\": {:.9},\n",
            "    \"average_occluded_pixels\": {:.3},\n",
            "    \"average_glow_pixels\": {:.3}\n",
            "  }},\n",
            "  \"machine\": {{\n",
            "    \"wall_seconds\": {:.6},\n",
            "    \"render_speed_vs_realtime\": {:.6},\n",
            "    \"memory_start_working_set_bytes\": {},\n",
            "    \"memory_end_working_set_bytes\": {},\n",
            "    \"memory_peak_working_set_bytes\": {}\n",
            "  }}\n",
            "}}\n"
        ),
        json_escape(&args.run_id),
        unix_now(),
        json_escape(manifest_claim(&args.scene_mode)),
        json_escape(&video_path.display().to_string()),
        json_escape(&visual_path.display().to_string()),
        json_escape(&state_path.display().to_string()),
        args.width,
        args.height,
        args.fps,
        args.duration,
        frame_count,
        json_escape(&args.video_encoder),
        args.crf,
        json_escape(&args.bitrate),
        args.render_threads,
        args.state_log_every,
        match &args.audio {
            Some(path) => format!("\"{}\"", json_escape(&path.display().to_string())),
            None => "null".to_string(),
        },
        args.sample_rate,
        args.audio.is_some(),
        args.audio.is_some() && args.mux_audio,
        json_escape(&args.scene_mode),
        json_escape(&args.palette),
        scene_motifs_json(&args.scene_mode),
        avg_fog,
        avg_occluded,
        avg_glow,
        wall_seconds,
        args.duration / wall_seconds.max(0.000_001),
        memory_start.0,
        memory_end.0,
        memory_start.1.max(memory_end.1)
    )
    .map_err(|e| format!("manifest write failed: {e}"))?;
    file.flush()
        .map_err(|e| format!("manifest flush failed: {e}"))?;
    Ok(())
}

fn slug(value: &str) -> String {
    let mut clean = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            clean.push(ch);
        } else if ch.is_whitespace() {
            clean.push('_');
        }
    }
    if clean.is_empty() {
        "truevision_weird_occlusion_rs".to_string()
    } else {
        clean
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
