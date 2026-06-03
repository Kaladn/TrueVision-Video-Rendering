# TrueVision Document State Movie

## Purpose

Treat a document as visual state over time.

```text
document page
-> page frame
-> cell state
-> glyph/state candidates
-> replayed page surface
```

This is the pure TrueVision lane for the "document is slow video" idea.
AnchorWorks may later consume approved meaning/state packets, but AnchorWorks
is not required to witness, replay, or surface the page state.

## Names

```text
AnchorWorks document_film
  page/frame visual evidence metadata for AW document intake

AnchorWorks answer_surface
  meaning-side count/path speech rendering

TrueVision document_state_movie
  visual-state witness/replay/surface lane for page frames
```

## State Loop

```text
Witness:
  page frames become TrueVision cell-state frames

Profile:
  stored luma/edge/color cells can produce glyph pattern candidates

Plan:
  not owned by this first tool

Replay:
  cell-state chunks reconstruct page surfaces

Surface:
  optional PNG output is derived display only
```

## Boundaries

```text
No OCR authority.
No AnchorWorks runtime dependency.
No raw page/frame retention by default.
No generated media as evidence.
No answer authority inside TrueVision.
Unknown glyphs remain unknown.
```

## Player Compatibility

The emitted manifest contains `cell_state` chunks and familiar playback
metadata, so the external State AV Player can treat document-state movies as
normal TrueVision state sources later.

## Law

```text
TrueVision witnesses the page.
TrueVision replays the state.
AnchorWorks interprets meaning later.
SecureCore gates action later.
```
