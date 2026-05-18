# TrueVision Synthetic State Media Formula Report

## Claim

A 5-second no-sound walking-person scene was generated as COD/TrueVision-shaped cell-state vectors, then replayed as video from those vectors.

## Boundary

```text
This is synthetic state media.
It is not evidence.
It is not prompt video.
It is a declared scene formula rendered into the same 16:9 cell-state shape used by COD/TrueVision capture.
```

## Formula

```text
SceneState(t)
  -> sky_layer + horizon_line + field_layer
  -> walking_person_actor(x(t), walk_phase(t), limb_phase(t))
  -> 90x160 addressed cells
  -> 16-feature cell vectors
  -> replayable COD/TrueVision bundle
```

## Outputs

- Run ID: `person_field_walk_5s_state_media`
- Run dir: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_walk_5s_state_media`
- Manifest: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_walk_5s_state_media\person_field_walk_5s_state_media_manifest.json`
- Summary: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_walk_5s_state_media\person_field_walk_5s_state_media_summary.json`
- Records: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_walk_5s_state_media\person_field_walk_5s_state_media_records.jsonl`
- Lossless replay: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_walk_5s_state_media\replay\person_field_walk_5s_state_media_cell_rgb_replay_lossless_ffv1.mkv`
- Preview replay: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_walk_5s_state_media\replay\person_field_walk_5s_state_media_cell_rgb_replay_preview_mp4v.mp4`

## Hardware Used

```json
{
  "python": "3.11.0",
  "platform": "Windows-10-10.0.26200-SP0",
  "processor": "AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD",
  "compute_path": "CPU numpy/OpenCV state generation plus CPU OpenCV VideoWriter encode",
  "gpu_acceleration_used": false,
  "cpu_logical": 32,
  "cpu_physical": 16,
  "ram_total_bytes": 66094223360,
  "ram_available_bytes_at_start": 43088883712,
  "gpu_adapters_detected": [
    {
      "Name": "AMD Radeon(TM) Graphics",
      "AdapterRAM": 2147483648,
      "DriverVersion": "32.0.21030.2001"
    },
    {
      "Name": "Intel(R) Arc(TM) Pro B70 Graphics",
      "AdapterRAM": 2147479552,
      "DriverVersion": "32.0.101.8517"
    }
  ]
}
```

## Timing

```json
{
  "started_at_utc": "2026-05-18T09:09:30.083Z",
  "completed_at_utc": "2026-05-18T09:09:31.178Z",
  "total_seconds": 1.095818,
  "state_generation_seconds": 0.589919,
  "replay_seconds": 0.505899,
  "frames": 45,
  "process_memory_start": {
    "rss_bytes": 306524160,
    "vms_bytes": 1053777920
  },
  "process_memory_end": {
    "rss_bytes": 319127552,
    "vms_bytes": 1077387264
  }
}
```

## No Audio

`audio_saved=false`; no audio stream is generated or muxed.
