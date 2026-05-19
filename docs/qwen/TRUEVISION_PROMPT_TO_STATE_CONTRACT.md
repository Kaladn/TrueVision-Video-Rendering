# TrueVision Prompt-To-State Contract

Qwen is not the authority. Qwen is a draft translator.

## Required Behavior

```text
Input: human AV prompt + project context + schema
Output: JSON draft only
No markdown
No prose
No tool execution claims
No evidence claims
```

## Required JSON Shape

```json
{
  "request_kind": "truevision_state_media_draft",
  "scene": {
    "name": "thin audio river"
  },
  "renderer": "edge_audio_river",
  "media": {
    "audio_path": "D:/music/song.wav",
    "sync_to_audio": true
  },
  "timeline": {
    "duration_seconds": 180.0,
    "fps": 12
  },
  "visual_parameters": {
    "river_height_ratio": 0.24,
    "black_field": true,
    "no_glyphs": true
  },
  "safety_boundary": {
    "generated_state_media": true,
    "evidence": false
  }
}
```

## WAV Rule

For WAV-driven videos, prefer:

```text
renderer=edge_audio_river
sync_to_audio=true
use audio_probe_duration when duration is unknown
use audio_analyze_levels to find peaks, valleys, rising energy, and section energy
use audio_extract_features before preview/full render when feature data is needed
use template_from_audio_signals when the output should be random geometry or pattern-library driven
```

## Repair Rule

If validation fails, the adapter sends only:

```json
{
  "validation_errors": ["..."],
  "repair_instruction": "Return corrected JSON only.",
  "original_context": {}
}
```

Qwen must return a corrected JSON object only.
