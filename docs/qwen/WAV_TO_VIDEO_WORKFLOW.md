# WAV To Video Workflow

Yes, TrueVision Generation Lab can make videos from WAV files.

## Direct Renderer Path

```powershell
cd D:\TrueVision_Generation_Lab
python scripts\truevision_edge_audio_river.py `
  --audio "D:\path\to\song.wav" `
  --output-root "outputs\wav_river" `
  --run-id "song_river" `
  --fps 12 `
  --width 1920 `
  --height 1080
```

## Adapter Path

```text
1. audio_probe_duration
2. audio_analyze_levels
3. audio_extract_features when band features are needed
4. template_from_audio_signals for geometry-driven visuals
5. prompt_to_state_adapter for human style/theme intent
6. template_create or template_save
7. video_render_preview
8. video_prepare_full_render
9. video_execute_full_render after human confirmation
```

## Recommended Visual Language

```text
black field
thin river
sound-level reactions
low band moves body
mid band moves color flow
high band adds shimmer
beat blooms intensity
no lettering
no glyphs
program stamp in letterbox band only when requested
```

## Signal Mapping

```text
peaks -> pulses, flashes, rings, random geometry spawn
valleys -> slow drift, dimming, holds, calmer camera
rising energy -> expansion, camera push, brighter color pressure
falling energy -> contraction, cooling, less motion
section energy -> scene intensity and transition pressure
```
