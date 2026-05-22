# Document State Reader

The document-state reader treats a document as visual state over time.

```text
document
-> page frames
-> glyph cells
-> glyph state records
-> document state read
-> optional derived text
```

One page is one frame. Glyphs are observed visual marks. Text is derived only
when the stored glyph state supports it.

## Boundary

```text
No OCR authority.
No raw string as truth.
No AnchorWorks runtime dependency.
No lexicon mutation during reads.
No lifetime-count mutation during reads.
Unknown glyphs remain unknown.
```

## Runtime Shape

```text
truevision_runtime/document_state/
  contracts.py       state packet constructors
  document_video.py  one-page-per-frame document video packets
  glyph_lexicon.py   read-only approved glyph matcher
  lifetime_counts.py read-only glyph lifetime context
  state_reader.py    page-frame glyph state reader
```

## Principle

```text
Machines do not need to see like people.
They need stable, inspectable state they can replay.
```

The reader is built for deterministic recall: the same visual glyph cells,
lexicon, and lifetime context produce the same state hashes.
