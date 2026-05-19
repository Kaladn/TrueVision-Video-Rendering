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
2. audio_extract_features
3. prompt_to_state_adapter
4. template_create
5. template_save
6. video_render_preview
7. video_prepare_full_render
8. video_execute_full_render after human confirmation
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
