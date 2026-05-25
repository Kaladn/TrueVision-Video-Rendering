# YouTube Source Surface Safe Ops

This document defines how a YouTube page can be used as an approved visual
source surface for Elemental Learning Intake without turning TrueVision into a
general browser bot.

## Rule

```text
Display IDs observe.
Button IDs require approval.
Forbidden controls never bind.
Receipts close the loop.
```

```text
A completed macro is not a completed capture.
A verified video-state receipt is a completed capture.
```

Navigation must go through the browser address bar, not YouTube search:

```text
approved URL
-> canonical watch URL
-> focus browser address bar
-> paste URL
-> Enter
-> wait for video page
-> verify URL/title/duration
-> capture
```

Playlist/query noise such as `list=...` is stripped before the address-bar
navigation step. The approved source is the canonical video ID.

Large videos use bounded section sampling:

```text
one-hour video
-> four section samples
-> profile and purge each sample
-> aggregate only compact behavior recipes
```

Default one-hour sample starts:

```text
00:07:24
00:22:24
00:37:24
00:52:24
```

## Purpose

The Learning Intake Module may use a YouTube page as a visual teacher source
only after the operator approves the source. The system records the source
surface state, capture plan, and receipt. It does not roam, download, comment,
subscribe, or make account actions.

## Surface Contract

Machine-readable contract:

```text
presets/learning_intake/youtube_source_surface_v1.json
```

The contract contains:

```text
display_ids
button_ids
forbidden_controls
preflight_checks
receipt_fields
queue_binding
```

## Display IDs

Display IDs are observable regions or values. They do not perform actions.

```text
yt.display.page_url
yt.display.search_query
yt.display.player_region
yt.display.title
yt.display.channel
yt.display.elapsed_time
yt.display.duration
yt.display.fullscreen_state
```

Allowed use:

```text
source identity
operator-visible search context
player/capture framing
title/channel receipt metadata
time alignment
layout preflight
```

## Button IDs

Button IDs are not free actions. They are only allowed when the operator or an
approved harness explicitly confirms the step and the receipt target exists.

```text
yt.button.play_pause
yt.button.seek_to_start
yt.button.fullscreen
yt.button.settings_speed
```

Allowed use:

```text
play/pause alignment
seek-to-start alignment
fullscreen capture setup
approved playback-speed selection
```

## Forbidden Controls

These must not be bound by this project:

```text
like
dislike
subscribe
comment
share
download
upload
notifications
account menu
recommendation links
external ads
comment input
```

## Preflight

Before capture:

```text
source approved by operator
canonical watch URL opened through browser address bar
resolved URL matches approved video ID
source title recorded if visible
duration detected from the loaded video page
player region identified
capture resolution and grid declared
audio setting declared
raw frame policy declared
retention intent declared
no forbidden controls requested
```

## Receipt

A safe source-surface receipt should record:

```text
surface_id
intake_id
element_id
source URL or note
resolved URL
visible title
visible channel
detected duration
visual state record count
not gray screen
not error page
profile created
teacher chunks purged
operator approval
display IDs observed
button IDs requested
button IDs approved
capture plan hash
receipt hash
```

## Boundary

```text
This is source-surface discipline.
It is not browser autonomy.
```
