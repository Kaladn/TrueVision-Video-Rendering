# TrueVision Full-Power Single Frame Report

## Claim

One clean frame was generated from a detailed scene state, sampled into COD/TrueVision 16-channel cell vectors, then reconstructed using more than RGB means.

## Channels Used

- `rgb_mean_r/g/b`
- `rgb_std_r/g/b`
- `luma_std`
- `texture_energy`
- `edge_density`
- `motion_energy`
- `delta_luma_abs`
- `saturation_mean`

## Boundary

```text
Synthetic state media.
Not evidence.
Not prompt generation.
State transition data is used through previous/current luma delta and motion channels.
```

## Outputs

- State frame: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_clean_frame_full_power\person_field_clean_frame_full_power_state_full_power.png`
- Source reference: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_clean_frame_full_power\person_field_clean_frame_full_power_source_reference.png`
- Cell state: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_clean_frame_full_power\person_field_clean_frame_full_power_cell_state.npz`
- Manifest: `D:\arc_solver_clean\cod_616\data\truevision_generated\person_field_clean_frame_full_power\person_field_clean_frame_full_power_manifest.json`

## Hardware

```json
{
  "python": "3.11.0",
  "platform": "Windows-10-10.0.26200-SP0",
  "processor": "AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD",
  "compute_path": "CPU numpy/OpenCV high-detail source render, cell-state sampling, and full-power state replay",
  "gpu_acceleration_used": false,
  "cpu_logical": 32,
  "cpu_physical": 16,
  "ram_total_bytes": 66094223360,
  "ram_available_bytes": 42599968768,
  "process_rss_bytes": 319016960,
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
