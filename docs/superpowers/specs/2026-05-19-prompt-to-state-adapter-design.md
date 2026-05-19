# Prompt To State Adapter Design

## Goal

Build a model-neutral adapter that turns human audio/video prompts into validated TrueVision AV state JSON without trusting the model endpoint.

## Boundary

```text
Model drafts.
Validator decides.
AV policy executes.
Renderer renders.
Manifest and receipts prove what happened.
```

The runtime must not depend on Qwen specifically. Qwen, Codex, OpenAI, Claude, Gemini, local Ollama, or another local endpoint can all sit behind the same adapter as long as they return draft JSON.

## Pipeline

```text
human prompt
+ project context
+ schema
+ examples
+ allowed AV fields
+ current template/artifact context
        ↓
model draft JSON
        ↓
schema validator
        ↓
repair loop if invalid
        ↓
canonical TrueVision AV state JSON
        ↓
AV tool runner
        ↓
manifest + receipt + artifact
```

## Trust Rules

- The app never trusts model output directly.
- The app trusts only validated state JSON.
- The generator receives only validated state JSON.
- Every tool request still passes through AV policy.
- Chat-origin tool calls stay unconfirmed.
- Full render execution and destructive actions require explicit human confirmation.
- Generated media is synthetic state media, not evidence.

## WAV Video Path

WAV files are first-class inputs for audio-reactive video generation.

```text
WAV file
-> audio_probe_duration
-> audio_analyze_levels
-> audio_extract_features
-> state pattern library
-> PromptToStateAdapter draft
-> schema validation
-> template_from_audio_signals when geometry is signal-driven
-> template_create/template_save
-> video_render_preview
-> video_prepare_full_render
-> video_execute_full_render after confirmation
```

The current renderer target for WAV-driven visuals is `edge_audio_river`. It can produce simple black-field color-river visuals from audio bands and lyric/theme hints.

For random geometry and richer visual language, the adapter should prefer `audio_geometry_field` templates that reference named state patterns. The pattern library starts with pulse rings, deterministic random shards, quiet valley drift, rising energy expansion, and high-energy edge shimmer.

## Runtime Files

```text
truevision_runtime/llm_adapter/
  prompt_to_state_adapter.py
  prompt_context_builder.py
  schema_validator.py

truevision_runtime/av_tools/
  av_tool_registry.py
  av_tool_policy.py
  av_tool_runner.py
  av_tool_receipts.py
```

## Done Means

- A prompt can be wrapped into a model-neutral adapter context.
- A model draft can be parsed as JSON.
- Invalid JSON/state receives validation errors only.
- A repaired draft can become canonical state JSON.
- WAV features can be extracted through an AV-only tool.
- Tool calls write receipts.
- The test suite passes.
