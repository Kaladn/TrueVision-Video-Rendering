use std::collections::VecDeque;
use std::env;
use std::ffi::c_void;
use std::fs::{create_dir_all, read_dir, File};
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

const FEATURE_COUNT: usize = 16;
const RGB_R: usize = 0;
const RGB_G: usize = 1;
const RGB_B: usize = 2;
const LUMA_MEAN: usize = 9;
const DELTA_LUMA_ABS: usize = 12;
const MOTION_ENERGY: usize = 15;

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

#[derive(Clone)]
struct Args {
    run_dir: PathBuf,
    output_dir: PathBuf,
    duration: f64,
    capture_fps: f64,
    target_fps: f64,
    resolution: (usize, usize),
    crf: u8,
    encoder: String,
    motion_mode: String,
    smoothing: String,
    temporal_radius: usize,
    recursion_depth: usize,
    crop_grid: Option<(usize, usize)>,
    max_cached_chunks: usize,
}

#[derive(Clone, Copy)]
struct CropRegion {
    x: usize,
    y: usize,
    cols: usize,
    rows: usize,
}

impl CropRegion {
    fn full(grid_rows: usize, grid_cols: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            cols: grid_cols,
            rows: grid_rows,
        }
    }

    fn centered(
        grid_rows: usize,
        grid_cols: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows > grid_rows || cols > grid_cols {
            return Err(format!(
                "crop grid {}x{} must fit inside source grid {}x{}",
                cols, rows, grid_cols, grid_rows
            ));
        }
        Ok(Self {
            x: (grid_cols - cols) / 2,
            y: (grid_rows - rows) / 2,
            cols,
            rows,
        })
    }

    fn label(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"cols\":{},\"rows\":{}}}",
            self.x, self.y, self.cols, self.rows
        )
    }
}

#[derive(Clone, Copy, Default)]
struct FrameSolveStats {
    confidence_sum: f64,
    confidence_samples: usize,
    low_confidence_cells: usize,
}

impl FrameSolveStats {
    fn add(&mut self, other: FrameSolveStats) {
        self.confidence_sum += other.confidence_sum;
        self.confidence_samples += other.confidence_samples;
        self.low_confidence_cells += other.low_confidence_cells;
    }

    fn average_confidence(&self) -> f64 {
        if self.confidence_samples == 0 {
            1.0
        } else {
            self.confidence_sum / self.confidence_samples as f64
        }
    }
}

struct SegmentField {
    left_index: usize,
    right_index: usize,
    grid_rows: usize,
    grid_cols: usize,
    dx: Vec<f32>,
    dy: Vec<f32>,
    confidence: Vec<f32>,
    stable_weight: Vec<f32>,
    drift: Vec<f32>,
    stats: FrameSolveStats,
}

struct ChunkMeta {
    path: PathBuf,
    start_index: usize,
    frame_count: usize,
    grid_rows: usize,
    grid_cols: usize,
}

struct ChunkData {
    index: usize,
    frame_numbers: Vec<u32>,
    cells: Vec<f32>,
    grid_rows: usize,
    grid_cols: usize,
}

struct ChunkCache {
    metas: Vec<ChunkMeta>,
    loaded: VecDeque<ChunkData>,
    max_cached_chunks: usize,
    peak_cached_chunks: usize,
    chunk_loads: usize,
}

impl ChunkCache {
    fn new(metas: Vec<ChunkMeta>, max_cached_chunks: usize) -> Result<Self, String> {
        if metas.is_empty() {
            return Err("no chunk metadata found".to_string());
        }
        if max_cached_chunks == 0 {
            return Err("max-cached-chunks must be positive".to_string());
        }
        Ok(Self {
            metas,
            loaded: VecDeque::new(),
            max_cached_chunks,
            peak_cached_chunks: 0,
            chunk_loads: 0,
        })
    }

    fn source_frame_count(&self) -> usize {
        self.metas
            .last()
            .map(|meta| meta.start_index + meta.frame_count)
            .unwrap_or(0)
    }

    fn meta_index_for_source_index(&self, source_index: usize) -> Result<usize, String> {
        self.metas
            .iter()
            .position(|meta| {
                source_index >= meta.start_index
                    && source_index < meta.start_index + meta.frame_count
            })
            .ok_or_else(|| format!("source index out of range: {source_index}"))
    }

    fn load_chunk(&mut self, meta_index: usize) -> Result<&ChunkData, String> {
        if let Some(position) = self
            .loaded
            .iter()
            .position(|chunk| chunk.index == meta_index)
        {
            let chunk = self.loaded.remove(position).expect("loaded chunk missing");
            self.loaded.push_back(chunk);
            return Ok(self.loaded.back().expect("loaded chunk disappeared"));
        }
        let meta = &self.metas[meta_index];
        let chunk = read_chunk_data(&meta.path, meta_index)?;
        self.loaded.push_back(chunk);
        self.chunk_loads += 1;
        while self.loaded.len() > self.max_cached_chunks {
            self.loaded.pop_front();
        }
        self.peak_cached_chunks = self.peak_cached_chunks.max(self.loaded.len());
        Ok(self.loaded.back().expect("loaded chunk disappeared"))
    }

    fn copy_source_frame_region(
        &mut self,
        source_index: usize,
        crop: CropRegion,
        out: &mut Vec<f32>,
    ) -> Result<u32, String> {
        let meta_index = self.meta_index_for_source_index(source_index)?;
        let local = source_index - self.metas[meta_index].start_index;
        let chunk = self.load_chunk(meta_index)?;
        let cells_per_frame = chunk.grid_rows * chunk.grid_cols * FEATURE_COUNT;
        let frame_start = local * cells_per_frame;
        out.clear();
        out.resize(crop.rows * crop.cols * FEATURE_COUNT, 0.0);
        for row in 0..crop.rows {
            let source_start =
                frame_start + ((crop.y + row) * chunk.grid_cols + crop.x) * FEATURE_COUNT;
            let source_end = source_start + crop.cols * FEATURE_COUNT;
            let target_start = row * crop.cols * FEATURE_COUNT;
            let target_end = target_start + crop.cols * FEATURE_COUNT;
            out[target_start..target_end].copy_from_slice(&chunk.cells[source_start..source_end]);
        }
        Ok(chunk.frame_numbers[local])
    }
}

fn read_u32(reader: &mut BufReader<File>) -> Result<u32, String> {
    let mut buf = [0_u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read u32 failed: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_chunk_header(path: &PathBuf, start_index: usize) -> Result<ChunkMeta, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|e| format!("open chunk failed {}: {e}", path.display()))?,
    );
    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("read chunk magic failed: {e}"))?;
    if &magic != b"TVCELL01" {
        return Err(format!("bad chunk magic: {}", path.display()));
    }
    let frame_count = read_u32(&mut reader)? as usize;
    let grid_rows = read_u32(&mut reader)? as usize;
    let grid_cols = read_u32(&mut reader)? as usize;
    let feature_count = read_u32(&mut reader)? as usize;
    if feature_count != FEATURE_COUNT {
        return Err(format!(
            "expected {FEATURE_COUNT} features, got {feature_count}: {}",
            path.display()
        ));
    }
    Ok(ChunkMeta {
        path: path.clone(),
        start_index,
        frame_count,
        grid_rows,
        grid_cols,
    })
}

fn read_chunk_data(path: &PathBuf, index: usize) -> Result<ChunkData, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|e| format!("open chunk failed {}: {e}", path.display()))?,
    );
    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("read chunk magic failed: {e}"))?;
    if &magic != b"TVCELL01" {
        return Err(format!("bad chunk magic: {}", path.display()));
    }
    let frame_count = read_u32(&mut reader)? as usize;
    let grid_rows = read_u32(&mut reader)? as usize;
    let grid_cols = read_u32(&mut reader)? as usize;
    let feature_count = read_u32(&mut reader)? as usize;
    if feature_count != FEATURE_COUNT {
        return Err(format!(
            "expected {FEATURE_COUNT} features, got {feature_count}: {}",
            path.display()
        ));
    }
    let mut frame_numbers = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        frame_numbers.push(read_u32(&mut reader)?);
    }
    let value_count = frame_count * grid_rows * grid_cols * FEATURE_COUNT;
    let mut bytes = vec![0_u8; value_count * 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("read chunk values failed: {e}"))?;
    let mut cells = Vec::with_capacity(value_count);
    for raw in bytes.chunks_exact(4) {
        cells.push(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
    }
    Ok(ChunkData {
        index,
        frame_numbers,
        cells,
        grid_rows,
        grid_cols,
    })
}

fn discover_chunks(run_dir: &PathBuf) -> Result<Vec<ChunkMeta>, String> {
    let cell_dir = run_dir.join("cell_state_native");
    let mut paths: Vec<PathBuf> = read_dir(&cell_dir)
        .map_err(|e| format!("read cell_state_native failed {}: {e}", cell_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("tvcells"))
        .collect();
    paths.sort();
    let mut metas = Vec::with_capacity(paths.len());
    let mut start_index = 0_usize;
    for path in paths {
        let meta = read_chunk_header(&path, start_index)?;
        start_index += meta.frame_count;
        metas.push(meta);
    }
    Ok(metas)
}

fn catmull_rom(a: f32, b: f32, c: f32, d: f32, u: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    0.5 * ((2.0 * b)
        + (-a + c) * u
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * u2
        + (-a + 3.0 * b - 3.0 * c + d) * u3)
}

fn smoothstep(alpha: f32) -> f32 {
    let u = alpha.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn cell_feature(frame: &[f32], grid_cols: usize, gy: usize, gx: usize, feature: usize) -> f32 {
    frame[(gy * grid_cols + gx) * FEATURE_COUNT + feature]
}

fn sample_feature_nearest(
    frame: &[f32],
    grid_rows: usize,
    grid_cols: usize,
    gy: i32,
    gx: i32,
    feature: usize,
) -> f32 {
    let y = gy.clamp(0, grid_rows.saturating_sub(1) as i32) as usize;
    let x = gx.clamp(0, grid_cols.saturating_sub(1) as i32) as usize;
    cell_feature(frame, grid_cols, y, x, feature)
}

fn sample_feature_bilinear(
    frame: &[f32],
    grid_rows: usize,
    grid_cols: usize,
    y: f32,
    x: f32,
    feature: usize,
) -> f32 {
    let y = y.clamp(0.0, grid_rows.saturating_sub(1) as f32);
    let x = x.clamp(0.0, grid_cols.saturating_sub(1) as f32);
    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(grid_rows.saturating_sub(1));
    let x1 = (x0 + 1).min(grid_cols.saturating_sub(1));
    let fy = y - y0 as f32;
    let fx = x - x0 as f32;
    let a = cell_feature(frame, grid_cols, y0, x0, feature);
    let b = cell_feature(frame, grid_cols, y0, x1, feature);
    let c = cell_feature(frame, grid_cols, y1, x0, feature);
    let d = cell_feature(frame, grid_cols, y1, x1, feature);
    let top = a * (1.0 - fx) + b * fx;
    let bottom = c * (1.0 - fx) + d * fx;
    top * (1.0 - fy) + bottom * fy
}

fn estimate_local_shift(
    a: &[f32],
    b: &[f32],
    grid_rows: usize,
    grid_cols: usize,
    gy: usize,
    gx: usize,
) -> (f32, f32, f32) {
    let cell = (gy * grid_cols + gx) * FEATURE_COUNT;
    let base_a = a[cell + LUMA_MEAN];
    let base_b = b[cell + LUMA_MEAN];
    let motion = a[cell + MOTION_ENERGY]
        .max(b[cell + MOTION_ENERGY])
        .max(a[cell + DELTA_LUMA_ABS])
        .max(b[cell + DELTA_LUMA_ABS]);
    let base_score = (base_a - base_b).abs();
    if motion < 2.0 && base_score < 4.0 {
        return (0.0, 0.0, 0.96);
    }

    let mut best_dx = 0_i32;
    let mut best_dy = 0_i32;
    let mut best_score = base_score;
    for dy in -1_i32..=1 {
        for dx in -1_i32..=1 {
            let shifted = sample_feature_nearest(
                b,
                grid_rows,
                grid_cols,
                gy as i32 + dy,
                gx as i32 + dx,
                LUMA_MEAN,
            );
            let score = (base_a - shifted).abs() + ((dx.abs() + dy.abs()) as f32 * 0.65);
            if score < best_score {
                best_score = score;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    let confidence = (1.0 - best_score / 72.0).clamp(0.18, 0.98);
    (best_dx as f32, best_dy as f32, confidence)
}

fn build_segment_field(
    a: &[f32],
    b: &[f32],
    left_index: usize,
    right_index: usize,
    grid_rows: usize,
    grid_cols: usize,
) -> SegmentField {
    let cell_count = grid_rows * grid_cols;
    let mut field = SegmentField {
        left_index,
        right_index,
        grid_rows,
        grid_cols,
        dx: vec![0.0; cell_count],
        dy: vec![0.0; cell_count],
        confidence: vec![1.0; cell_count],
        stable_weight: vec![0.0; cell_count],
        drift: vec![0.0; a.len()],
        stats: FrameSolveStats::default(),
    };

    for gy in 0..grid_rows {
        for gx in 0..grid_cols {
            let cell_index = gy * grid_cols + gx;
            let cell = cell_index * FEATURE_COUNT;
            let (dx, dy, confidence) = estimate_local_shift(a, b, grid_rows, grid_cols, gy, gx);
            let motion = a[cell + MOTION_ENERGY]
                .max(b[cell + MOTION_ENERGY])
                .max(a[cell + DELTA_LUMA_ABS])
                .max(b[cell + DELTA_LUMA_ABS]);
            let luma_delta = (a[cell + LUMA_MEAN] - b[cell + LUMA_MEAN]).abs();
            let motion_stability = 1.0 - (motion / 18.0).clamp(0.0, 1.0);
            let luma_stability = 1.0 - (luma_delta / 48.0).clamp(0.0, 1.0);

            field.dx[cell_index] = dx;
            field.dy[cell_index] = dy;
            field.confidence[cell_index] = confidence;
            field.stable_weight[cell_index] = (motion_stability * luma_stability).clamp(0.0, 1.0);
            field.stats.confidence_sum += confidence as f64;
            field.stats.confidence_samples += 1;
            if confidence < 0.45 {
                field.stats.low_confidence_cells += 1;
            }

            for feature in 0..FEATURE_COUNT {
                let index = cell + feature;
                field.drift[index] = b[index] - a[index];
            }
        }
    }

    field
}

fn render_segment_field(
    a: &[f32],
    b: &[f32],
    field: &SegmentField,
    alpha: f32,
    out: &mut Vec<f32>,
) {
    let grid_rows = field.grid_rows;
    let grid_cols = field.grid_cols;
    let eased = smoothstep(alpha);
    out.resize(a.len(), 0.0);

    for gy in 0..grid_rows {
        for gx in 0..grid_cols {
            let cell_index = gy * grid_cols + gx;
            let cell = cell_index * FEATURE_COUNT;
            let confidence = field.confidence[cell_index];
            let stable_weight = field.stable_weight[cell_index];
            let dx = field.dx[cell_index] * confidence;
            let dy = field.dy[cell_index] * confidence;

            for feature in 0..FEATURE_COUNT {
                let index = cell + feature;
                let direct = a[index] + field.drift[index] * eased;
                let from_a = sample_feature_bilinear(
                    a,
                    grid_rows,
                    grid_cols,
                    gy as f32 - dy * eased,
                    gx as f32 - dx * eased,
                    feature,
                );
                let from_b = sample_feature_bilinear(
                    b,
                    grid_rows,
                    grid_cols,
                    gy as f32 + dy * (1.0 - eased),
                    gx as f32 + dx * (1.0 - eased),
                    feature,
                );
                let moved = from_a * (1.0 - eased) + from_b * eased;
                let confidence_mix = moved * confidence + direct * (1.0 - confidence);
                let value =
                    confidence_mix * (1.0 - stable_weight * 0.35) + direct * (stable_weight * 0.35);
                let lo = a[index].min(b[index]).min(direct);
                let hi = a[index].max(b[index]).max(direct);
                out[index] = value.clamp(lo, hi);
            }
        }
    }
}

fn solve_midpoint_state(
    a: &[f32],
    b: &[f32],
    grid_rows: usize,
    grid_cols: usize,
    out: &mut Vec<f32>,
) -> FrameSolveStats {
    out.resize(a.len(), 0.0);
    let mut stats = FrameSolveStats::default();
    for gy in 0..grid_rows {
        for gx in 0..grid_cols {
            let cell = (gy * grid_cols + gx) * FEATURE_COUNT;
            let (dx, dy, confidence) = estimate_local_shift(a, b, grid_rows, grid_cols, gy, gx);
            stats.confidence_sum += confidence as f64;
            stats.confidence_samples += 1;
            if confidence < 0.45 {
                stats.low_confidence_cells += 1;
            }
            let stable_from_a = a[cell + MOTION_ENERGY] <= b[cell + MOTION_ENERGY];
            for feature in 0..FEATURE_COUNT {
                let index = cell + feature;
                let from_a = sample_feature_bilinear(
                    a,
                    grid_rows,
                    grid_cols,
                    gy as f32 - dy * 0.5,
                    gx as f32 - dx * 0.5,
                    feature,
                );
                let from_b = sample_feature_bilinear(
                    b,
                    grid_rows,
                    grid_cols,
                    gy as f32 + dy * 0.5,
                    gx as f32 + dx * 0.5,
                    feature,
                );
                let midpoint = (from_a + from_b) * 0.5;
                let stable = if stable_from_a { a[index] } else { b[index] };
                let value = midpoint * confidence + stable * (1.0 - confidence);
                out[index] = value.clamp(a[index].min(b[index]), a[index].max(b[index]));
            }
        }
    }
    stats
}

fn solve_recursive_midpoint_state(
    a: &[f32],
    b: &[f32],
    alpha: f32,
    depth: usize,
    grid_rows: usize,
    grid_cols: usize,
    out: &mut Vec<f32>,
) -> FrameSolveStats {
    if alpha <= 0.0001 {
        out.clear();
        out.extend_from_slice(a);
        return FrameSolveStats {
            confidence_sum: 1.0,
            confidence_samples: 1,
            low_confidence_cells: 0,
        };
    }
    if alpha >= 0.9999 {
        out.clear();
        out.extend_from_slice(b);
        return FrameSolveStats {
            confidence_sum: 1.0,
            confidence_samples: 1,
            low_confidence_cells: 0,
        };
    }

    let mut midpoint = Vec::<f32>::new();
    let mut stats = solve_midpoint_state(a, b, grid_rows, grid_cols, &mut midpoint);
    if depth == 0 || (alpha - 0.5).abs() <= (0.5_f32 / (1_u32 << depth.min(20)) as f32) {
        out.clear();
        out.extend_from_slice(&midpoint);
        return stats;
    }

    let child_stats = if alpha < 0.5 {
        solve_recursive_midpoint_state(
            a,
            &midpoint,
            alpha * 2.0,
            depth - 1,
            grid_rows,
            grid_cols,
            out,
        )
    } else {
        solve_recursive_midpoint_state(
            &midpoint,
            b,
            (alpha - 0.5) * 2.0,
            depth - 1,
            grid_rows,
            grid_cols,
            out,
        )
    };
    stats.add(child_stats);
    stats
}

fn estimate_global_shift(
    a: &[f32],
    b: &[f32],
    grid_rows: usize,
    grid_cols: usize,
) -> (i32, i32, f32) {
    let max_shift = 14_i32;
    let shift_step = 2_usize;
    let sample_step_y = (grid_rows / 45).max(4);
    let sample_step_x = (grid_cols / 80).max(4);
    let mut best = (0_i32, 0_i32, f32::MAX);
    for dy in (-max_shift..=max_shift).step_by(shift_step) {
        for dx in (-max_shift..=max_shift).step_by(shift_step) {
            let mut sum = 0.0_f32;
            let mut count = 0_usize;
            for gy in (max_shift as usize..grid_rows.saturating_sub(max_shift as usize))
                .step_by(sample_step_y)
            {
                for gx in (max_shift as usize..grid_cols.saturating_sub(max_shift as usize))
                    .step_by(sample_step_x)
                {
                    let by = gy as i32 + dy;
                    let bx = gx as i32 + dx;
                    let av = cell_feature(a, grid_cols, gy, gx, LUMA_MEAN);
                    let bv = sample_feature_nearest(b, grid_rows, grid_cols, by, bx, LUMA_MEAN);
                    sum += (av - bv).abs();
                    count += 1;
                }
            }
            if count > 0 {
                let score = sum / count as f32;
                if score < best.2 {
                    best = (dx, dy, score);
                }
            }
        }
    }
    best
}

fn interpolate_state(
    cache: &mut ChunkCache,
    target_time: f64,
    capture_fps: f64,
    grid_rows: usize,
    grid_cols: usize,
    motion_mode: &str,
    temporal_radius: usize,
    recursion_depth: usize,
    crop: CropRegion,
    buffers: &mut [Vec<f32>; 7],
    segment_field: &mut Option<SegmentField>,
) -> Result<(u32, u32, f32, i32, i32, usize, FrameSolveStats), String> {
    let last_index = cache.source_frame_count().saturating_sub(1);
    let source_position = (target_time * capture_fps).clamp(0.0, last_index as f64);
    let left = source_position.floor() as usize;
    let right = (left + 1).min(last_index);
    let alpha = (source_position - left as f64) as f32;
    let n0 = cache.copy_source_frame_region(left.saturating_sub(1), crop, &mut buffers[0])?;
    let n1 = cache.copy_source_frame_region(left, crop, &mut buffers[1])?;
    let n2 = cache.copy_source_frame_region(right, crop, &mut buffers[2])?;
    let _n3 = cache.copy_source_frame_region((right + 1).min(last_index), crop, &mut buffers[3])?;
    buffers[4].resize(buffers[1].len(), 0.0);
    let mut solve_stats = FrameSolveStats {
        confidence_sum: 1.0,
        confidence_samples: 1,
        low_confidence_cells: 0,
    };
    if left == right {
        let source = buffers[1].clone();
        buffers[4].copy_from_slice(&source);
    } else if motion_mode == "adaptive-shift" {
        let (dx, dy, _score) =
            estimate_global_shift(&buffers[1], &buffers[2], grid_rows, grid_cols);
        let ax = (alpha * dx as f32).round() as i32;
        let ay = (alpha * dy as f32).round() as i32;
        let bx = ((1.0 - alpha) * dx as f32).round() as i32;
        let by = ((1.0 - alpha) * dy as f32).round() as i32;
        for gy in 0..grid_rows {
            for gx in 0..grid_cols {
                let cell = (gy * grid_cols + gx) * FEATURE_COUNT;
                let motion = buffers[1][cell + MOTION_ENERGY]
                    .max(buffers[2][cell + MOTION_ENERGY])
                    .max(buffers[1][cell + DELTA_LUMA_ABS])
                    .max(buffers[2][cell + DELTA_LUMA_ABS]);
                let use_shift = motion > 2.0 && (dx.abs() + dy.abs()) > 0;
                for feature in 0..FEATURE_COUNT {
                    let value = if use_shift {
                        let from_a = sample_feature_nearest(
                            &buffers[1],
                            grid_rows,
                            grid_cols,
                            gy as i32 - ay,
                            gx as i32 - ax,
                            feature,
                        );
                        let from_b = sample_feature_nearest(
                            &buffers[2],
                            grid_rows,
                            grid_cols,
                            gy as i32 + by,
                            gx as i32 + bx,
                            feature,
                        );
                        from_a * (1.0 - alpha) + from_b * alpha
                    } else {
                        buffers[1][cell + feature] * (1.0 - alpha)
                            + buffers[2][cell + feature] * alpha
                    };
                    let lo = buffers[0][cell + feature]
                        .min(buffers[1][cell + feature])
                        .min(buffers[2][cell + feature])
                        .min(buffers[3][cell + feature]);
                    let hi = buffers[0][cell + feature]
                        .max(buffers[1][cell + feature])
                        .max(buffers[2][cell + feature])
                        .max(buffers[3][cell + feature]);
                    buffers[4][cell + feature] = value.clamp(lo, hi);
                }
            }
        }
        return Ok((n1.max(n0), n2, alpha, dx, dy, 0, solve_stats));
    } else if motion_mode == "recursive-midpoint" {
        let left_state = buffers[1].clone();
        let right_state = buffers[2].clone();
        solve_stats = solve_recursive_midpoint_state(
            &left_state,
            &right_state,
            alpha,
            recursion_depth,
            grid_rows,
            grid_cols,
            &mut buffers[4],
        );
        return Ok((n1.max(n0), n2, alpha, 0, 0, 0, solve_stats));
    } else if motion_mode == "segment-field" {
        let needs_field = segment_field
            .as_ref()
            .map(|field| {
                field.left_index != left
                    || field.right_index != right
                    || field.grid_rows != grid_rows
                    || field.grid_cols != grid_cols
            })
            .unwrap_or(true);
        if needs_field {
            *segment_field = Some(build_segment_field(
                &buffers[1],
                &buffers[2],
                left,
                right,
                grid_rows,
                grid_cols,
            ));
        }
        let field = segment_field
            .as_ref()
            .ok_or_else(|| "segment field cache unavailable".to_string())?;
        let (source_buffers, output_buffers) = buffers.split_at_mut(4);
        render_segment_field(
            &source_buffers[1],
            &source_buffers[2],
            field,
            alpha,
            &mut output_buffers[0],
        );
        return Ok((n1.max(n0), n2, alpha, 0, 0, 0, field.stats));
    } else if motion_mode == "temporal-map" {
        let radius = temporal_radius.max(1);
        let before = left.saturating_sub(radius);
        let after = (right + radius).min(last_index);
        cache.copy_source_frame_region(before, crop, &mut buffers[5])?;
        cache.copy_source_frame_region(after, crop, &mut buffers[6])?;
        let span = (after.saturating_sub(before)).max(1) as f32;
        let causal_alpha = ((source_position as f32 - before as f32) / span).clamp(0.0, 1.0);
        for cell_index in 0..(grid_rows * grid_cols) {
            let cell = cell_index * FEATURE_COUNT;
            let motion = buffers[1][cell + MOTION_ENERGY]
                .max(buffers[2][cell + MOTION_ENERGY])
                .max(buffers[1][cell + DELTA_LUMA_ABS])
                .max(buffers[2][cell + DELTA_LUMA_ABS]);
            let causal_weight = if motion > 12.0 {
                0.18
            } else if motion > 4.0 {
                0.25
            } else {
                0.34
            };
            for feature in 0..FEATURE_COUNT {
                let index = cell + feature;
                let a = buffers[0][index];
                let b = buffers[1][index];
                let c = buffers[2][index];
                let d = buffers[3][index];
                let far_a = buffers[5][index];
                let far_b = buffers[6][index];
                let spline = catmull_rom(a, b, c, d, alpha);
                let causal = far_a * (1.0 - causal_alpha) + far_b * causal_alpha;
                let value = spline * (1.0 - causal_weight) + causal * causal_weight;
                let lo = a.min(b).min(c).min(d).min(far_a).min(far_b);
                let hi = a.max(b).max(c).max(d).max(far_a).max(far_b);
                buffers[4][index] = value.clamp(lo, hi);
            }
        }
        return Ok((n1.max(n0), n2, alpha, 0, 0, radius, solve_stats));
    } else if motion_mode == "linear" {
        for index in 0..buffers[4].len() {
            buffers[4][index] = buffers[1][index] * (1.0 - alpha) + buffers[2][index] * alpha;
        }
    } else {
        for index in 0..buffers[4].len() {
            let a = buffers[0][index];
            let b = buffers[1][index];
            let c = buffers[2][index];
            let d = buffers[3][index];
            let lo = a.min(b).min(c).min(d);
            let hi = a.max(b).max(c).max(d);
            buffers[4][index] = catmull_rom(a, b, c, d, alpha).clamp(lo, hi);
        }
    }
    Ok((n1.max(n0), n2, alpha, 0, 0, 0, solve_stats))
}

fn smooth_state_rgb(state: &[f32], grid_rows: usize, grid_cols: usize, scratch: &mut Vec<f32>) {
    scratch.resize(state.len(), 0.0);
    scratch.copy_from_slice(state);
    for gy in 0..grid_rows {
        for gx in 0..grid_cols {
            let cell = (gy * grid_cols + gx) * FEATURE_COUNT;
            let motion = state[cell + MOTION_ENERGY].max(state[cell + DELTA_LUMA_ABS]);
            let weight_center = if motion > 8.0 { 0.76 } else { 0.56 };
            let weight_neighbor = (1.0 - weight_center) / 4.0;
            for feature in [RGB_R, RGB_G, RGB_B] {
                let mut value = state[cell + feature] * weight_center;
                let mut weight = weight_center;
                if gx > 0 {
                    value += state[(gy * grid_cols + (gx - 1)) * FEATURE_COUNT + feature]
                        * weight_neighbor;
                    weight += weight_neighbor;
                }
                if gx + 1 < grid_cols {
                    value += state[(gy * grid_cols + (gx + 1)) * FEATURE_COUNT + feature]
                        * weight_neighbor;
                    weight += weight_neighbor;
                }
                if gy > 0 {
                    value += state[((gy - 1) * grid_cols + gx) * FEATURE_COUNT + feature]
                        * weight_neighbor;
                    weight += weight_neighbor;
                }
                if gy + 1 < grid_rows {
                    value += state[((gy + 1) * grid_cols + gx) * FEATURE_COUNT + feature]
                        * weight_neighbor;
                    weight += weight_neighbor;
                }
                scratch[cell + feature] = value / weight;
            }
        }
    }
}

fn render_rgb_frame(
    state: &[f32],
    grid_rows: usize,
    grid_cols: usize,
    width: usize,
    height: usize,
    out: &mut [u8],
) {
    let cell_w = width / grid_cols;
    let cell_h = height / grid_rows;
    for gy in 0..grid_rows {
        for gx in 0..grid_cols {
            let cell_index = (gy * grid_cols + gx) * FEATURE_COUNT;
            let r = state[cell_index + RGB_R].round().clamp(0.0, 255.0) as u8;
            let g = state[cell_index + RGB_G].round().clamp(0.0, 255.0) as u8;
            let b = state[cell_index + RGB_B].round().clamp(0.0, 255.0) as u8;
            for py in (gy * cell_h)..((gy + 1) * cell_h) {
                let row = py * width * 3;
                for px in (gx * cell_w)..((gx + 1) * cell_w) {
                    let idx = row + px * 3;
                    out[idx] = r;
                    out[idx + 1] = g;
                    out[idx + 2] = b;
                }
            }
        }
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn parse_pair(value: &str, name: &str) -> Result<(usize, usize), String> {
    let parts: Vec<&str> = value.split('x').collect();
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
        return Err(format!("{name} must be positive"));
    }
    Ok((width, height))
}

fn parse_args() -> Result<Args, String> {
    let mut raw = env::args().skip(1);
    let mut args = Args {
        run_dir: PathBuf::new(),
        output_dir: PathBuf::new(),
        duration: 0.0,
        capture_fps: 9.0,
        target_fps: 60.0,
        resolution: (2560, 1440),
        crf: 18,
        encoder: "libx264".to_string(),
        motion_mode: "catmull".to_string(),
        smoothing: "off".to_string(),
        temporal_radius: 6,
        recursion_depth: 3,
        crop_grid: None,
        max_cached_chunks: 3,
    };
    while let Some(flag) = raw.next() {
        let value = raw
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--run-dir" => args.run_dir = PathBuf::from(value),
            "--output-dir" => args.output_dir = PathBuf::from(value),
            "--duration" => {
                args.duration = value
                    .parse::<f64>()
                    .map_err(|_| "bad duration".to_string())?
            }
            "--capture-fps" => {
                args.capture_fps = value
                    .parse::<f64>()
                    .map_err(|_| "bad capture-fps".to_string())?
            }
            "--target-fps" => {
                args.target_fps = value
                    .parse::<f64>()
                    .map_err(|_| "bad target-fps".to_string())?
            }
            "--resolution" => args.resolution = parse_pair(&value, "resolution")?,
            "--crf" => args.crf = value.parse::<u8>().map_err(|_| "bad crf".to_string())?,
            "--encoder" => args.encoder = value,
            "--motion-mode" => args.motion_mode = value,
            "--smoothing" => args.smoothing = value,
            "--temporal-radius" => {
                args.temporal_radius = value
                    .parse::<usize>()
                    .map_err(|_| "bad temporal-radius".to_string())?
                    .max(1)
            }
            "--recursion-depth" => {
                args.recursion_depth = value
                    .parse::<usize>()
                    .map_err(|_| "bad recursion-depth".to_string())?
            }
            "--crop-grid" => {
                let (cols, rows) = parse_pair(&value, "crop-grid")?;
                args.crop_grid = Some((cols, rows));
            }
            "--max-cached-chunks" => {
                args.max_cached_chunks = value
                    .parse::<usize>()
                    .map_err(|_| "bad max-cached-chunks".to_string())?
            }
            _ => return Err(format!("unknown flag: {flag}")),
        }
    }
    if args.run_dir.as_os_str().is_empty() {
        return Err("--run-dir is required".to_string());
    }
    if args.output_dir.as_os_str().is_empty() {
        return Err("--output-dir is required".to_string());
    }
    if args.duration <= 0.0 {
        return Err("--duration must be positive".to_string());
    }
    if args.capture_fps <= 0.0 || args.target_fps <= 0.0 {
        return Err("fps values must be positive".to_string());
    }
    Ok(args)
}

fn ffmpeg_encode_args(args: &Args, video_path: &PathBuf) -> Vec<String> {
    let mut command = vec![
        "-y".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-s".to_string(),
        format!("{}x{}", args.resolution.0, args.resolution.1),
        "-r".to_string(),
        format!("{}", args.target_fps),
        "-i".to_string(),
        "-".to_string(),
        "-an".to_string(),
    ];
    match args.encoder.as_str() {
        "h264_qsv" | "hevc_qsv" | "av1_qsv" => {
            command.extend([
                "-vf".to_string(),
                "format=nv12".to_string(),
                "-c:v".to_string(),
                args.encoder.clone(),
                "-global_quality".to_string(),
                format!("{}", args.crf),
                "-look_ahead".to_string(),
                "0".to_string(),
            ]);
        }
        "h264_amf" | "hevc_amf" | "av1_amf" => {
            command.extend([
                "-vf".to_string(),
                "format=nv12".to_string(),
                "-c:v".to_string(),
                args.encoder.clone(),
                "-quality".to_string(),
                "speed".to_string(),
                "-qp_i".to_string(),
                format!("{}", args.crf),
                "-qp_p".to_string(),
                format!("{}", args.crf),
            ]);
        }
        "h264_d3d12va" | "hevc_d3d12va" | "av1_d3d12va" | "h264_vulkan" | "hevc_vulkan"
        | "av1_vulkan" => {
            command.extend([
                "-vf".to_string(),
                "format=nv12".to_string(),
                "-c:v".to_string(),
                args.encoder.clone(),
            ]);
        }
        _ => {
            command.extend([
                "-c:v".to_string(),
                "libx264".to_string(),
                "-pix_fmt".to_string(),
                "yuv420p".to_string(),
                "-crf".to_string(),
                format!("{}", args.crf),
            ]);
        }
    }
    command.push(video_path.display().to_string());
    command
}

fn timeline_rule_for_mode(motion_mode: &str) -> &'static str {
    match motion_mode {
        "segment-field" => "segment_transition_field_inside_source_duration_not_append_at_end",
        "recursive-midpoint" => "recursive_midpoint_fill_inside_source_duration_not_append_at_end",
        "temporal-map" => "temporal_map_fill_inside_source_duration_not_append_at_end",
        "linear" => "linear_fill_inside_source_duration_not_append_at_end",
        _ => "catmull_rom_fill_inside_source_duration_not_append_at_end",
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    create_dir_all(&args.output_dir).map_err(|e| format!("create output dir failed: {e}"))?;
    let run_id = args
        .run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "run-dir must have a final path component".to_string())?
        .to_string();
    let metas = discover_chunks(&args.run_dir)?;
    let first = metas.first().ok_or_else(|| "no chunks found".to_string())?;
    let grid_rows = first.grid_rows;
    let grid_cols = first.grid_cols;
    let crop = match args.crop_grid {
        Some((cols, rows)) => CropRegion::centered(grid_rows, grid_cols, rows, cols)?,
        None => CropRegion::full(grid_rows, grid_cols),
    };
    let work_rows = crop.rows;
    let work_cols = crop.cols;
    let (width, height) = args.resolution;
    if width % work_cols != 0 || height % work_rows != 0 {
        return Err("resolution must divide evenly by render/crop grid".to_string());
    }
    let mut cache = ChunkCache::new(metas, args.max_cached_chunks)?;
    let output_frames = (args.duration * args.target_fps).round().max(1.0) as usize;
    let video_path = args.output_dir.join(format!(
        "{}_trueframegen_stream_rs_{}fps.mp4",
        run_id,
        args.target_fps.round() as u32
    ));
    let manifest_path = args.output_dir.join(format!(
        "{}_trueframegen_stream_rs_{}fps_manifest.json",
        run_id,
        args.target_fps.round() as u32
    ));
    let trace_path = args.output_dir.join(format!(
        "{}_trueframegen_stream_rs_{}fps_trace.jsonl",
        run_id,
        args.target_fps.round() as u32
    ));
    let report_path = args.output_dir.join(format!(
        "{}_trueframegen_stream_rs_{}fps_report.md",
        run_id,
        args.target_fps.round() as u32
    ));

    let ffmpeg_args = ffmpeg_encode_args(&args, &video_path);
    let mut ffmpeg = Command::new("ffmpeg")
        .args(&ffmpeg_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg start failed: {e}"))?;

    let mut stdin = ffmpeg
        .stdin
        .take()
        .ok_or_else(|| "ffmpeg stdin unavailable".to_string())?;
    let mut trace = File::create(&trace_path).map_err(|e| format!("trace open failed: {e}"))?;
    let mut buffers = [
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<f32>::new(),
    ];
    let mut frame = vec![0_u8; width * height * 3];
    let mut smooth_state = Vec::<f32>::new();
    let started = Instant::now();
    let trace_every = args.target_fps.round().max(1.0) as usize;
    let mut solve_confidence_sum = 0.0_f64;
    let mut solve_confidence_frames = 0_usize;
    let mut low_confidence_ranges: Vec<(usize, usize)> = Vec::new();
    let mut open_low_confidence_start: Option<usize> = None;
    let mut segment_field_cache: Option<SegmentField> = None;
    let timeline_rule = timeline_rule_for_mode(&args.motion_mode);
    for output_index in 0..output_frames {
        let target_time = output_index as f64 / args.target_fps;
        let (left_anchor, right_anchor, alpha, dx, dy, temporal_radius_used, solve_stats) =
            interpolate_state(
                &mut cache,
                target_time,
                args.capture_fps,
                work_rows,
                work_cols,
                &args.motion_mode,
                args.temporal_radius,
                args.recursion_depth,
                crop,
                &mut buffers,
                &mut segment_field_cache,
            )?;
        let frame_confidence = solve_stats.average_confidence();
        solve_confidence_sum += frame_confidence;
        solve_confidence_frames += 1;
        if frame_confidence < 0.55 {
            if open_low_confidence_start.is_none() {
                open_low_confidence_start = Some(output_index);
            }
        } else if let Some(start) = open_low_confidence_start.take() {
            low_confidence_ranges.push((start, output_index.saturating_sub(1)));
        }
        if args.smoothing == "rgb-neighbor" {
            smooth_state_rgb(&buffers[4], work_rows, work_cols, &mut smooth_state);
            render_rgb_frame(
                &smooth_state,
                work_rows,
                work_cols,
                width,
                height,
                &mut frame,
            );
        } else {
            render_rgb_frame(&buffers[4], work_rows, work_cols, width, height, &mut frame);
        }
        stdin
            .write_all(&frame)
            .map_err(|e| format!("ffmpeg write failed: {e}"))?;
        if output_index % trace_every == 0 || output_index + 1 == output_frames {
            writeln!(
                trace,
                "{{\"output_frame\":{},\"target_time_seconds\":{:.6},\"anchors\":[{},{}],\"alpha\":{:.6},\"global_shift_cells\":[{},{}],\"motion_mode\":\"{}\",\"smoothing\":\"{}\",\"temporal_radius_used\":{},\"recursion_depth\":{},\"frame_confidence\":{:.6},\"low_confidence_cells\":{},\"cached_chunks\":{},\"peak_cached_chunks\":{},\"not_appended\":true,\"renderer\":\"rust\"}}",
                output_index,
                target_time,
                left_anchor,
                right_anchor,
                alpha,
                dx,
                dy,
                json_escape(&args.motion_mode),
                json_escape(&args.smoothing),
                temporal_radius_used,
                args.recursion_depth,
                frame_confidence,
                solve_stats.low_confidence_cells,
                cache.loaded.len(),
                cache.peak_cached_chunks
            )
            .map_err(|e| format!("trace write failed: {e}"))?;
        }
    }
    if let Some(start) = open_low_confidence_start.take() {
        low_confidence_ranges.push((start, output_frames.saturating_sub(1)));
    }
    drop(stdin);
    let output = ffmpeg
        .wait_with_output()
        .map_err(|e| format!("ffmpeg wait failed: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let elapsed = started.elapsed().as_secs_f64();
    let video_bytes = std::fs::metadata(&video_path)
        .map(|meta| meta.len())
        .unwrap_or(0);
    let average_confidence = if solve_confidence_frames == 0 {
        1.0
    } else {
        solve_confidence_sum / solve_confidence_frames as f64
    };
    let low_confidence_json = format!(
        "[{}]",
        low_confidence_ranges
            .iter()
            .map(|(start, end)| {
                format!(
                    "{{\"start_frame\":{},\"end_frame\":{},\"start_seconds\":{:.6},\"end_seconds\":{:.6}}}",
                    start,
                    end,
                    *start as f64 / args.target_fps,
                    *end as f64 / args.target_fps
                )
            })
            .collect::<Vec<String>>()
            .join(",")
    );
    let source_frames_used =
        ((args.duration * args.capture_fps).ceil() as usize + 1).min(cache.source_frame_count());
    let generated_frame_count = output_frames.saturating_sub(source_frames_used);
    let (working_set_bytes, peak_working_set_bytes) = process_memory_snapshot();
    std::fs::write(
        &manifest_path,
        format!(
            "{{\n  \"schema_version\": 1,\n  \"kind\": \"trueframegen_stream_rs_manifest\",\n  \"source_run_id\": \"{}\",\n  \"law\": \"TrueVision records. Rust TrueFrameGen streams bounded in-between state.\",\n  \"source\": {{\"run_dir\": \"{}\", \"source_frames\": {}, \"source_frames_used\": {}, \"capture_fps\": {}, \"grid_shape\": [{}, {}]}},\n  \"upsample\": {{\"target_fps\": {}, \"duration_seconds\": {:.6}, \"output_frames\": {}, \"generated_frame_count\": {}, \"timeline_rule\": \"{}\", \"motion_mode\": \"{}\", \"smoothing\": \"{}\", \"temporal_radius\": {}, \"recursion_depth\": {}, \"crop_region\": {}, \"average_confidence\": {:.6}, \"low_confidence_frame_ranges\": {}, \"state_dump_written\": false}},\n  \"streaming\": {{\"max_cached_chunks\": {}, \"peak_cached_chunks\": {}, \"chunks_loaded\": {}, \"elapsed_generation_seconds\": {:.6}, \"renderer\": \"rust\", \"encoder\": \"{}\", \"process_working_set_bytes\": {}, \"process_peak_working_set_bytes\": {}}},\n  \"outputs\": {{\"video_mp4\": \"{}\", \"video_bytes\": {}, \"trace_jsonl\": \"{}\", \"report_md\": \"{}\"}}\n}}\n",
            json_escape(&run_id),
            json_escape(&args.run_dir.display().to_string()),
            cache.source_frame_count(),
            source_frames_used,
            args.capture_fps,
            grid_rows,
            grid_cols,
            args.target_fps,
            args.duration,
            output_frames,
            generated_frame_count,
            timeline_rule,
            json_escape(&args.motion_mode),
            json_escape(&args.smoothing),
            args.temporal_radius,
            args.recursion_depth,
            crop.label(),
            average_confidence,
            low_confidence_json,
            args.max_cached_chunks,
            cache.peak_cached_chunks,
            cache.chunk_loads,
            elapsed,
            json_escape(&args.encoder),
            working_set_bytes,
            peak_working_set_bytes,
            json_escape(&video_path.display().to_string()),
            video_bytes,
            json_escape(&trace_path.display().to_string()),
            json_escape(&report_path.display().to_string()),
        ),
    )
    .map_err(|e| format!("manifest write failed: {e}"))?;
    std::fs::write(
        &report_path,
        format!(
            "# {} Rust TrueFrameGen Stream Report\n\n- Duration: `{:.6}s`\n- Target FPS: `{}`\n- Output frames: `{}`\n- Motion mode: `{}`\n- Recursion depth: `{}`\n- Crop region: `{}`\n- Average confidence: `{:.6}`\n- Low-confidence frame ranges: `{}`\n- Peak cached chunks: `{}`\n- Chunks loaded: `{}`\n- Generation wall time: `{:.3}s`\n- Process working set: `{}`\n- Process peak working set: `{}`\n- Video: `{}`\n\n`{}`\n",
            run_id,
            args.duration,
            args.target_fps,
            output_frames,
            args.motion_mode,
            args.recursion_depth,
            crop.label(),
            average_confidence,
            low_confidence_json,
            cache.peak_cached_chunks,
            cache.chunk_loads,
            elapsed,
            working_set_bytes,
            peak_working_set_bytes,
            video_path.display(),
            timeline_rule
        ),
    )
    .map_err(|e| format!("report write failed: {e}"))?;
    println!(
        "{{\n  \"manifest_json\": \"{}\",\n  \"video_mp4\": \"{}\",\n  \"trace_jsonl\": \"{}\",\n  \"report_md\": \"{}\",\n  \"elapsed_generation_seconds\": {:.6}\n}}",
        json_escape(&manifest_path.display().to_string()),
        json_escape(&video_path.display().to_string()),
        json_escape(&trace_path.display().to_string()),
        json_escape(&report_path.display().to_string()),
        elapsed
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
