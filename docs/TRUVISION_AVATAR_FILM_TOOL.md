# TrueVision Avatar Film Tool

This tool preserves the TrueVision avatar as a reusable production asset while letting each film swap in new presentation material.

## Permanent Avatar Kit

Tracked avatar-only assets live here:

```text
truevision_runtime/rendering/avatar_assets/truvision_avatar_v1/
  manifest.json
  poses/
```

Do not put project slides, audio stems, narration text, or presentation-specific labels in that folder.

## Change Per Film

Each film should use a project JSON based on:

```text
templates/truvision_avatar_film_project.example.json
```

Change these per project:

```text
slides
audio
segments
system_labels
film_title
output_dir
output_name
```

Keep this unless intentionally replacing the avatar:

```text
avatar_pose_dir:
  D:/TrueVision_Generation_Lab/truevision_runtime/rendering/avatar_assets/truvision_avatar_v1/poses
```

## Render

```powershell
python scripts/render_truvision_avatar_film.py path\to\project.json
```

The renderer writes the final MP4 and render receipts into the configured `output_dir`.
