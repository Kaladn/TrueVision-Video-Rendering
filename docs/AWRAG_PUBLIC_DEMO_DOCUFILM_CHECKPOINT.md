# AWRAG Public Demo Docufilm Checkpoint

Date: 2026-06-18
Source system: AnchorWorks / AWRAG public reviewer demo
TrueVision lane: AnchorWorks code-system docufilm

## Purpose

This note gives TrueVision the correct documentary frame for the public AWRAG
demo. The demo should be visualized as a serious evidence engine slice, not as a
local-doc toy and not as the full private AnchorWorks system.

## Core Frame

```text
Public AWRAG demo
-> admitted dataset
-> dataset-local lexical values
-> public demo symbols
-> dataset-local relation counts
-> source coordinates
-> AWRAG-owned citations
-> evidence qualification receipts
-> evidence/coordinate packet
```

The public demo shows the evidence contract. It does not claim final production
speech rendering.

## Symbol Namespace Law

Public AWRAG symbols are demo-safe dataset-local implementation IDs:

```text
symbol_system: awrag_public_6b@1
symbol_bytes: 6
symbol_scope: dataset_local_demo_only
transferable: false
lifetime_allowed: false
anchorworks_lifetime_symbol_compatible: false
```

TrueVision should not present these symbols as the private AnchorWorks/Cortex
Evolved Systems symbol genome. They are not lifetime symbols. They are not
portable authority outside the dataset package.

## Private System Boundary

```text
Public AWRAG:
  six-byte demo symbol namespace
  dataset-local counts
  dataset-local citations
  reviewer-inspectable receipts

Private AnchorWorks:
  protected lifetime symbol genome
  proprietary symbol assignment
  system-specific memory authority
```

The visual story should keep these namespaces visibly separate.

## Current Completion State

Done:

- public repo clone/run path works
- dataset-local intake works
- dataset lexicon and counts are created beside the dataset
- coordinate index and citation JSONL are emitted
- query output is an evidence/coordinate packet
- evidence qualifier emits qualification receipts
- public six-byte symbol namespace is declared in code, lexicon, manifest, tests,
  and README
- demo surface is frozen for stabilization

Not claimed:

- final natural-language answer rendering
- LLM reasoning authority
- private lifetime memory
- private symbol genome export
- enterprise connector implementation inside the public demo

## Docufilm Shot Ideas

```text
Shot 1:
  Raw admitted dataset enters a local boundary.

Shot 2:
  Dataset-local lexicon forms, with public six-byte symbols stamped as
  demo-only.

Shot 3:
  Relation counts form beside the dataset, not in lifetime memory.

Shot 4:
  Coordinates and citations bind evidence back to source lines.

Shot 5:
  Evidence qualifier rejects nearby-but-not-answer-bearing text.

Shot 6:
  Final packet is shown as cited evidence, not final speech.

Shot 7:
  Private AnchorWorks symbol genome remains behind a separate sealed boundary.
```

## Law

```text
Public AWRAG proves the dataset-local evidence contract.
Private AnchorWorks keeps lifetime symbol authority.
TrueVision documents the boundary without blurring it.
```

