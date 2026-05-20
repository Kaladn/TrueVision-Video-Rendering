use std::env;
use std::ffi::c_void;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type Bool = i32;
type Dword = u32;
type Hbitmap = isize;
type Hdc = isize;
type Hgdiobj = isize;
type Hwnd = isize;
type Int = i32;
type Long = i32;
type Uint = u32;
type Word = u16;

#[repr(C)]
struct BitmapInfoHeader {
    bi_size: Dword,
    bi_width: Long,
    bi_height: Long,
    bi_planes: Word,
    bi_bit_count: Word,
    bi_compression: Dword,
    bi_size_image: Dword,
    bi_x_pels_per_meter: Long,
    bi_y_pels_per_meter: Long,
    bi_clr_used: Dword,
    bi_clr_important: Dword,
}

#[repr(C)]
struct RgbQuad {
    rgb_blue: u8,
    rgb_green: u8,
    rgb_red: u8,
    rgb_reserved: u8,
}

#[repr(C)]
struct BitmapInfo {
    bmi_header: BitmapInfoHeader,
    bmi_colors: [RgbQuad; 1],
}

#[link(name = "User32")]
unsafe extern "system" {
    fn GetDC(hwnd: Hwnd) -> Hdc;
    fn ReleaseDC(hwnd: Hwnd, hdc: Hdc) -> Int;
    fn GetSystemMetrics(n_index: Int) -> Int;
}

#[link(name = "Gdi32")]
unsafe extern "system" {
    fn CreateCompatibleDC(hdc: Hdc) -> Hdc;
    fn DeleteDC(hdc: Hdc) -> Bool;
    fn CreateCompatibleBitmap(hdc: Hdc, cx: Int, cy: Int) -> Hbitmap;
    fn SelectObject(hdc: Hdc, h: Hgdiobj) -> Hgdiobj;
    fn DeleteObject(ho: Hgdiobj) -> Bool;
    fn BitBlt(
        hdc: Hdc,
        x: Int,
        y: Int,
        cx: Int,
        cy: Int,
        hdc_src: Hdc,
        x1: Int,
        y1: Int,
        rop: Dword,
    ) -> Bool;
    fn StretchBlt(
        hdc_dest: Hdc,
        x_dest: Int,
        y_dest: Int,
        w_dest: Int,
        h_dest: Int,
        hdc_src: Hdc,
        x_src: Int,
        y_src: Int,
        w_src: Int,
        h_src: Int,
        rop: Dword,
    ) -> Bool;
    fn SetStretchBltMode(hdc: Hdc, mode: Int) -> Int;
    fn GetDIBits(
        hdc: Hdc,
        hbm: Hbitmap,
        start: Uint,
        c_lines: Uint,
        lpv_bits: *mut c_void,
        lpbmi: *mut BitmapInfo,
        usage: Uint,
    ) -> Int;
}

const SM_CXSCREEN: Int = 0;
const SM_CYSCREEN: Int = 1;
const SRCCOPY: Dword = 0x00CC_0020;
const BI_RGB: Dword = 0;
const DIB_RGB_COLORS: Uint = 0;
const COLORONCOLOR: Int = 3;

const FEATURE_NAMES: [&str; 16] = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
];

#[derive(Clone)]
struct Args {
    duration: f64,
    fps: f64,
    resolution: (usize, usize),
    grid: (usize, usize),
    region: Option<(i32, i32, i32, i32)>,
    output_root: PathBuf,
    run_id: String,
    start_delay: f64,
    chunk_frames: usize,
}

struct CaptureContext {
    screen_dc: Hdc,
    mem_dc: Hdc,
    bitmap: Hbitmap,
    width: usize,
    height: usize,
}

impl Drop for CaptureContext {
    fn drop(&mut self) {
        unsafe {
            DeleteObject(self.bitmap as Hgdiobj);
            DeleteDC(self.mem_dc);
            ReleaseDC(0, self.screen_dc);
        }
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
    if args.resolution.0 % args.grid.0 != 0 || args.resolution.1 % args.grid.1 != 0 {
        return Err("resolution must divide evenly by grid".to_string());
    }
    if args.start_delay > 0.0 {
        println!(
            "Starting native capture in {:.3} seconds.",
            args.start_delay
        );
        sleep(Duration::from_secs_f64(args.start_delay));
    }

    let run_dir = args.output_root.join(&args.run_id);
    let cell_dir = run_dir.join("cell_state_native");
    create_dir_all(&cell_dir).map_err(|e| format!("create run dir failed: {e}"))?;

    let records_path = run_dir.join(format!("{}_records.jsonl", args.run_id));
    let mut records = BufWriter::new(
        File::create(&records_path).map_err(|e| format!("records open failed: {e}"))?,
    );

    let source_region = source_region(&args);
    let ctx = CaptureContext::new(args.resolution.0, args.resolution.1)?;
    let mut bgra = vec![0_u8; args.resolution.0 * args.resolution.1 * 4];
    let mut previous_luma = vec![0_f32; args.grid.0 * args.grid.1];
    let mut have_previous = false;

    let mut chunk_frames: Vec<f32> =
        Vec::with_capacity(args.chunk_frames * args.grid.0 * args.grid.1 * FEATURE_NAMES.len());
    let mut chunk_numbers: Vec<u32> = Vec::with_capacity(args.chunk_frames);
    let mut chunks: Vec<ChunkMeta> = Vec::new();

    let started = Instant::now();
    let mut frame_number: u32 = 0;
    while started.elapsed().as_secs_f64() < args.duration {
        let frame_started = Instant::now();
        ctx.capture(source_region, &mut bgra)?;
        let screen_energy = build_cell_state(
            &bgra,
            &args,
            &mut previous_luma,
            &mut have_previous,
            &mut chunk_frames,
        );
        chunk_numbers.push(frame_number);

        let elapsed = started.elapsed().as_secs_f64();
        writeln!(
            records,
            "{{\"schema_version\":1,\"record_kind\":\"truevision_native_rs_frame_state\",\"run_id\":\"{}\",\"observed_at_utc\":\"{}\",\"frame_number\":{},\"elapsed_seconds\":{:.6},\"fps\":{},\"screen_energy\":{:.3},\"raw_frame_saved\":false,\"raw_grid_saved\":false}}",
            json_escape(&args.run_id),
            utc_timestamp(),
            frame_number,
            elapsed,
            args.fps,
            screen_energy
        )
        .map_err(|e| format!("records write failed: {e}"))?;

        if chunk_numbers.len() >= args.chunk_frames {
            flush_chunk(
                &args,
                &cell_dir,
                &mut chunks,
                &mut chunk_frames,
                &mut chunk_numbers,
            )?;
        }

        frame_number += 1;
        let target = Duration::from_secs_f64(1.0 / args.fps.max(0.001));
        let spent = frame_started.elapsed();
        if spent < target {
            sleep(target - spent);
        }
    }
    flush_chunk(
        &args,
        &cell_dir,
        &mut chunks,
        &mut chunk_frames,
        &mut chunk_numbers,
    )?;
    records
        .flush()
        .map_err(|e| format!("records flush failed: {e}"))?;

    let duration_seconds = started.elapsed().as_secs_f64();
    write_summary(
        &args,
        &run_dir,
        frame_number,
        duration_seconds,
        source_region,
    )?;
    write_manifest(
        &args,
        &run_dir,
        &records_path,
        &chunks,
        frame_number,
        duration_seconds,
        source_region,
    )?;

    println!(
        "{{\n  \"run_id\": \"{}\",\n  \"frames\": {},\n  \"duration_seconds\": {:.3},\n  \"run_dir\": \"{}\",\n  \"records_jsonl\": \"{}\",\n  \"summary_json\": \"{}\",\n  \"manifest_json\": \"{}\"\n}}",
        json_escape(&args.run_id),
        frame_number,
        duration_seconds,
        json_escape(&run_dir.display().to_string()),
        json_escape(&records_path.display().to_string()),
        json_escape(&run_dir.join(format!("{}_summary.json", args.run_id)).display().to_string()),
        json_escape(&run_dir.join(format!("{}_manifest.json", args.run_id)).display().to_string()),
    );
    Ok(())
}

impl CaptureContext {
    fn new(width: usize, height: usize) -> Result<Self, String> {
        unsafe {
            let screen_dc = GetDC(0);
            if screen_dc == 0 {
                return Err("GetDC failed".to_string());
            }
            let mem_dc = CreateCompatibleDC(screen_dc);
            if mem_dc == 0 {
                ReleaseDC(0, screen_dc);
                return Err("CreateCompatibleDC failed".to_string());
            }
            let bitmap = CreateCompatibleBitmap(screen_dc, width as i32, height as i32);
            if bitmap == 0 {
                DeleteDC(mem_dc);
                ReleaseDC(0, screen_dc);
                return Err("CreateCompatibleBitmap failed".to_string());
            }
            SelectObject(mem_dc, bitmap as Hgdiobj);
            SetStretchBltMode(mem_dc, COLORONCOLOR);
            Ok(Self {
                screen_dc,
                mem_dc,
                bitmap,
                width,
                height,
            })
        }
    }

    fn capture(&self, region: (i32, i32, i32, i32), out_bgra: &mut [u8]) -> Result<(), String> {
        let (left, top, source_w, source_h) = region;
        unsafe {
            let ok = if source_w == self.width as i32 && source_h == self.height as i32 {
                BitBlt(
                    self.mem_dc,
                    0,
                    0,
                    self.width as i32,
                    self.height as i32,
                    self.screen_dc,
                    left,
                    top,
                    SRCCOPY,
                )
            } else {
                StretchBlt(
                    self.mem_dc,
                    0,
                    0,
                    self.width as i32,
                    self.height as i32,
                    self.screen_dc,
                    left,
                    top,
                    source_w,
                    source_h,
                    SRCCOPY,
                )
            };
            if ok == 0 {
                return Err("screen blit failed".to_string());
            }
            let mut info = BitmapInfo {
                bmi_header: BitmapInfoHeader {
                    bi_size: std::mem::size_of::<BitmapInfoHeader>() as u32,
                    bi_width: self.width as i32,
                    bi_height: -(self.height as i32),
                    bi_planes: 1,
                    bi_bit_count: 32,
                    bi_compression: BI_RGB,
                    bi_size_image: 0,
                    bi_x_pels_per_meter: 0,
                    bi_y_pels_per_meter: 0,
                    bi_clr_used: 0,
                    bi_clr_important: 0,
                },
                bmi_colors: [RgbQuad {
                    rgb_blue: 0,
                    rgb_green: 0,
                    rgb_red: 0,
                    rgb_reserved: 0,
                }],
            };
            let lines = GetDIBits(
                self.mem_dc,
                self.bitmap,
                0,
                self.height as u32,
                out_bgra.as_mut_ptr() as *mut c_void,
                &mut info,
                DIB_RGB_COLORS,
            );
            if lines == 0 {
                return Err("GetDIBits failed".to_string());
            }
        }
        Ok(())
    }
}

fn build_cell_state(
    bgra: &[u8],
    args: &Args,
    previous_luma: &mut [f32],
    have_previous: &mut bool,
    output: &mut Vec<f32>,
) -> f32 {
    let (width, height) = args.resolution;
    let (grid_w, grid_h) = args.grid;
    let cell_w = width / grid_w;
    let cell_h = height / grid_h;
    let pixels_per_cell = (cell_w * cell_h) as f32;
    let mut screen_energy = 0.0_f32;

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let mut sum_r = 0.0_f32;
            let mut sum_g = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_r2 = 0.0_f32;
            let mut sum_g2 = 0.0_f32;
            let mut sum_b2 = 0.0_f32;
            let mut sum_l = 0.0_f32;
            let mut sum_l2 = 0.0_f32;
            let mut sum_sat = 0.0_f32;
            let mut sum_v = 0.0_f32;

            for y in (gy * cell_h)..((gy + 1) * cell_h) {
                let row = y * width * 4;
                for x in (gx * cell_w)..((gx + 1) * cell_w) {
                    let idx = row + x * 4;
                    let b = bgra[idx] as f32;
                    let g = bgra[idx + 1] as f32;
                    let r = bgra[idx + 2] as f32;
                    let maxc = r.max(g).max(b);
                    let minc = r.min(g).min(b);
                    let sat = if maxc > 0.0 {
                        ((maxc - minc) / maxc) * 255.0
                    } else {
                        0.0
                    };
                    let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                    sum_r += r;
                    sum_g += g;
                    sum_b += b;
                    sum_r2 += r * r;
                    sum_g2 += g * g;
                    sum_b2 += b * b;
                    sum_l += luma;
                    sum_l2 += luma * luma;
                    sum_sat += sat;
                    sum_v += maxc;
                }
            }

            let r_mean = sum_r / pixels_per_cell;
            let g_mean = sum_g / pixels_per_cell;
            let b_mean = sum_b / pixels_per_cell;
            let r_std = ((sum_r2 / pixels_per_cell) - r_mean * r_mean)
                .max(0.0)
                .sqrt();
            let g_std = ((sum_g2 / pixels_per_cell) - g_mean * g_mean)
                .max(0.0)
                .sqrt();
            let b_std = ((sum_b2 / pixels_per_cell) - b_mean * b_mean)
                .max(0.0)
                .sqrt();
            let luma_mean = sum_l / pixels_per_cell;
            let luma_std = ((sum_l2 / pixels_per_cell) - luma_mean * luma_mean)
                .max(0.0)
                .sqrt();
            let sat_mean = sum_sat / pixels_per_cell;
            let value_mean = sum_v / pixels_per_cell;
            let cell_index = gy * grid_w + gx;
            let delta_luma = if *have_previous {
                (luma_mean - previous_luma[cell_index]).abs()
            } else {
                0.0
            };
            previous_luma[cell_index] = luma_mean;
            screen_energy += delta_luma + luma_std * 0.05;

            output.extend_from_slice(&[
                r_mean, g_mean, b_mean, r_std, g_std, b_std, 0.0, sat_mean, value_mean, luma_mean,
                luma_std, sat_mean, delta_luma, 0.0, luma_std, delta_luma,
            ]);
        }
    }
    *have_previous = true;
    screen_energy
}

struct ChunkMeta {
    path: PathBuf,
    chunk_id: usize,
    frames: usize,
}

fn flush_chunk(
    args: &Args,
    cell_dir: &PathBuf,
    chunks: &mut Vec<ChunkMeta>,
    chunk_frames: &mut Vec<f32>,
    chunk_numbers: &mut Vec<u32>,
) -> Result<(), String> {
    if chunk_numbers.is_empty() {
        return Ok(());
    }
    let chunk_id = chunks.len();
    let path = cell_dir.join(format!("{}_cells_{:04}.tvcells", args.run_id, chunk_id));
    let mut file =
        BufWriter::new(File::create(&path).map_err(|e| format!("chunk open failed: {e}"))?);
    file.write_all(b"TVCELL01")
        .map_err(|e| format!("chunk write failed: {e}"))?;
    for value in [
        chunk_numbers.len() as u32,
        args.grid.1 as u32,
        args.grid.0 as u32,
        FEATURE_NAMES.len() as u32,
    ] {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| format!("chunk write failed: {e}"))?;
    }
    for number in chunk_numbers.iter() {
        file.write_all(&number.to_le_bytes())
            .map_err(|e| format!("chunk write failed: {e}"))?;
    }
    for value in chunk_frames.iter() {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| format!("chunk write failed: {e}"))?;
    }
    file.flush()
        .map_err(|e| format!("chunk flush failed: {e}"))?;
    chunks.push(ChunkMeta {
        path,
        chunk_id,
        frames: chunk_numbers.len(),
    });
    chunk_frames.clear();
    chunk_numbers.clear();
    Ok(())
}

fn write_summary(
    args: &Args,
    run_dir: &PathBuf,
    frame_count: u32,
    duration_seconds: f64,
    source_region: (i32, i32, i32, i32),
) -> Result<(), String> {
    let path = run_dir.join(format!("{}_summary.json", args.run_id));
    let text = format!(
        "{{\n  \"schema_version\": 1,\n  \"kind\": \"truevision_native_rs_summary\",\n  \"run_id\": \"{}\",\n  \"frame_count\": {},\n  \"duration_seconds\": {:.6},\n  \"geometry\": {{\n    \"source_shape\": [{}, {}],\n    \"frame_shape\": [{}, {}],\n    \"grid_shape\": [{}, {}],\n    \"capture_region\": [{}, {}, {}, {}]\n  }}\n}}\n",
        json_escape(&args.run_id),
        frame_count,
        duration_seconds,
        source_region.3,
        source_region.2,
        args.resolution.1,
        args.resolution.0,
        args.grid.1,
        args.grid.0,
        source_region.0,
        source_region.1,
        source_region.2,
        source_region.3,
    );
    std::fs::write(path, text).map_err(|e| format!("summary write failed: {e}"))
}

fn write_manifest(
    args: &Args,
    run_dir: &PathBuf,
    records_path: &PathBuf,
    chunks: &[ChunkMeta],
    frame_count: u32,
    duration_seconds: f64,
    source_region: (i32, i32, i32, i32),
) -> Result<(), String> {
    let path = run_dir.join(format!("{}_manifest.json", args.run_id));
    let mut chunk_lines = Vec::new();
    for chunk in chunks {
        chunk_lines.push(format!(
            "      {{\"chunk_id\": {}, \"path\": \"{}\", \"format\": \"tvcells_f32le_v1\", \"frames\": {}, \"grid_shape\": [{}, {}], \"feature_count\": {}}}",
            chunk.chunk_id,
            json_escape(&chunk.path.display().to_string()),
            chunk.frames,
            args.grid.1,
            args.grid.0,
            FEATURE_NAMES.len()
        ));
    }
    let feature_json = FEATURE_NAMES
        .iter()
        .map(|name| format!("\"{}\"", name))
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        "{{\n  \"schema_version\": 1,\n  \"record_kind\": \"truevision_native_rs_frame_state\",\n  \"run_id\": \"{}\",\n  \"created_at_utc\": \"{}\",\n  \"records_jsonl\": \"{}\",\n  \"config\": {{\n    \"duration_seconds\": {},\n    \"capture_fps\": {},\n    \"capture_resolution\": [{}, {}],\n    \"grid_size_xy\": [{}, {}],\n    \"capture_region\": [{}, {}, {}, {}],\n    \"cell_chunk_frames\": {}\n  }},\n  \"summary\": {{\"frame_count\": {}, \"duration_seconds\": {:.6}}},\n  \"cell_state\": {{\n    \"enabled\": true,\n    \"format\": \"tvcells_f32le_v1\",\n    \"feature_names\": [{}],\n    \"chunks\": [\n{}\n    ]\n  }},\n  \"boundary\": {{\n    \"raw_frame_saved\": false,\n    \"generated_media_is_evidence\": false,\n    \"notes\": \"Native Rust capture writes TrueVision cell state, not raw frames.\"\n  }}\n}}\n",
        json_escape(&args.run_id),
        utc_timestamp(),
        json_escape(&records_path.display().to_string()),
        args.duration,
        args.fps,
        args.resolution.0,
        args.resolution.1,
        args.grid.0,
        args.grid.1,
        source_region.0,
        source_region.1,
        source_region.2,
        source_region.3,
        args.chunk_frames,
        frame_count,
        duration_seconds,
        feature_json,
        chunk_lines.join(",\n"),
    );
    std::fs::write(path, text).map_err(|e| format!("manifest write failed: {e}"))
}

fn source_region(args: &Args) -> (i32, i32, i32, i32) {
    if let Some(region) = args.region {
        return region;
    }
    unsafe {
        (
            0,
            0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN),
        )
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut out = Args {
        duration: 60.0,
        fps: 9.0,
        resolution: (960, 540),
        grid: (160, 90),
        region: None,
        output_root: PathBuf::from(
            "E:\\TruEVision Generation\\library\\capture_units\\20_minute\\incoming",
        ),
        run_id: format!("truevision_rs_{}", timestamp_slug()),
        start_delay: 0.0,
        chunk_frames: 30,
    };
    while let Some(flag) = args.next() {
        let value = match flag.as_str() {
            "--duration"
            | "--fps"
            | "--resolution"
            | "--grid"
            | "--region"
            | "--output-root"
            | "--run-id"
            | "--start-delay"
            | "--cell-chunk-frames" => args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        };
        match flag.as_str() {
            "--duration" => out.duration = parse_f64(&value, "duration")?,
            "--fps" => out.fps = parse_f64(&value, "fps")?,
            "--resolution" => out.resolution = parse_pair(&value, "resolution")?,
            "--grid" => out.grid = parse_pair(&value, "grid")?,
            "--region" => out.region = Some(parse_region(&value)?),
            "--output-root" => out.output_root = PathBuf::from(value),
            "--run-id" => out.run_id = value,
            "--start-delay" => out.start_delay = parse_f64(&value, "start-delay")?,
            "--cell-chunk-frames" => {
                out.chunk_frames = value
                    .parse::<usize>()
                    .map_err(|_| "bad cell-chunk-frames".to_string())?
            }
            _ => {}
        }
    }
    Ok(out)
}

fn print_help() {
    println!(
        "truevision_capture_rs --duration 5 --fps 9 --resolution 2560x1440 --grid 640x360 --output-root <dir> --run-id <id>"
    );
}

fn parse_f64(value: &str, name: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("bad {name}: {value}"))
}

fn parse_pair(value: &str, name: &str) -> Result<(usize, usize), String> {
    let parts = value.split('x').collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err(format!("{name} must look like WIDTHxHEIGHT"));
    }
    let width = parts[0]
        .parse::<usize>()
        .map_err(|_| format!("bad {name} width"))?;
    let height = parts[1]
        .parse::<usize>()
        .map_err(|_| format!("bad {name} height"))?;
    if width == 0 || height == 0 {
        return Err(format!("{name} values must be positive"));
    }
    Ok((width, height))
}

fn parse_region(value: &str) -> Result<(i32, i32, i32, i32), String> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 4 {
        return Err("region must look like left,top,width,height".to_string());
    }
    let left = parts[0]
        .parse::<i32>()
        .map_err(|_| "bad region left".to_string())?;
    let top = parts[1]
        .parse::<i32>()
        .map_err(|_| "bad region top".to_string())?;
    let width = parts[2]
        .parse::<i32>()
        .map_err(|_| "bad region width".to_string())?;
    let height = parts[3]
        .parse::<i32>()
        .map_err(|_| "bad region height".to_string())?;
    if width <= 0 || height <= 0 {
        return Err("region width/height must be positive".to_string());
    }
    Ok((left, top, width, height))
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn timestamp_slug() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}

fn utc_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix_ms:{}", now.as_millis())
}
