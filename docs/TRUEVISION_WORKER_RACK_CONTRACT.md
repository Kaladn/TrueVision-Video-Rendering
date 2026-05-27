# TrueVision Worker Rack Contract

TrueVision is the logging organ.

Workers are small tools that inspect TrueVision logs and write compact,
reviewable artifacts. A worker is not the system, not the brain, and not a
backend.

```text
TrueVision logs.
Workers inspect near the organ that produced the logs.
SecureCore agents coordinate.
Operators approve.
Receipts prove.
```

## Core Law

```text
One worker.
One job.
One artifact.
One receipt.
No hidden work.
```

## Worker Packet

Every worker must declare this packet before it is considered active:

```text
worker_id
worker_name
input_lane
output_lane
allowed_reads
allowed_writes
forbidden_actions
receipt_required
operator_approval_required
status
```

Local chat-forged tools and workers are staged through:

```text
scripts/truevision_worker_forge.py
```

The forge writes manifests, append-only logs, and receipts. It chooses local
candidates only; it does not execute them.

Every run must answer:

```text
what_did_i_read
what_did_i_write
what_did_i_refuse
what_is_unknown
what_receipt_proves_this
```

## Hard Boundaries

Workers must not:

```text
promote truth
own meaning
silently guess
retain bulky source media by default
launch browsers unless explicitly operator-approved
delete source files unless explicitly operator-approved
turn generated media into evidence
become a general app, player, backend, or god bridge
```

Workers must:

```text
read only approved inputs
write only declared outputs
emit unknowns
emit rejection reasons
exit cleanly
write receipts when runtime state changes
```

## SecureCore Agent Boundary

Only agents transfer to SecureCore.

Workers stay close to the organs they inspect:

```text
TrueVision workers stay with TrueVision.
TrueAudio workers stay with TrueAudio.
TrueFrameGen workers stay with TrueFrameGen.
SecureCore receives only coordinating/reasoning/gating agents.
```

SecureCore agent candidates must use SecureCore's exact live-agent manifest
shape. A transfer package may stage those agents before integration, but a
staged package is not live activation.

Zero-tolerance rules:

```text
no prompt-only agents
no missing manifest fields
no fake entrypoint hashes
no worker migration into SecureCore
no policy authority in organ workers
no promotion-ready claim before SecureCore tests pass
```

## Recognition Worker Stack

The recognition rack is candidate-first. It learns reusable state signatures,
not truth labels.

```text
TrueVision log
-> bounds_worker
-> region_worker
-> shape_worker
-> glyph_worker
-> context_worker
-> persistence_worker
-> external_label_support_worker
-> consensus_worker
-> review_packet_worker
-> signature_writer_worker
```

### 1. Source Intake Worker

Job: collect images, clips, captions, references, source lists, and operator
notes into a source manifest.

Reads:

```text
source folder
file list
URL list
manual operator labels
```

Writes:

```text
source_manifest.jsonl
```

Does not:

```text
edit
tag
generate
decide meaning
```

Output fields:

```text
source_id
path_or_url
source_type
duration_or_dimensions
created_at
operator_note
status
```

### 2. Plate Bounds Worker

Job: find usable image/video content bounds.

Reads:

```text
source_manifest.jsonl
```

Writes:

```text
plate_bounds.jsonl
```

Does not:

```text
crop permanently
generate
stylize
```

Output fields:

```text
source_id
content_bounds
aspect_ratio
safe_center
black_bar_detected
face_or_subject_region_candidate
confidence
unknowns
```

### 3. Geometry Shape Worker

Job: extract shape structure without naming objects.

Reads:

```text
image samples
video frame samples
TrueVision cell/state logs
```

Writes:

```text
geometry_shape_profile.jsonl
```

Does not:

```text
label objects
claim identity
promote recognition
```

Output fields:

```text
source_id
region_id
contours
dominant_lines
curve_pressure
symmetry
vanishing_point_candidate
foreground_midground_background_candidate
shape_stability
geometry_signature
```

### 4. Geography Context Worker

Job: identify place/context candidates without claiming location truth.

Geometry is shape. Geography is place context.

Reads:

```text
scene metadata
frame metadata
road/sign/sky/building context candidates
GPS or map data only when explicitly available
```

Writes:

```text
geography_context_profile.jsonl
```

Does not:

```text
claim real-world location without GPS/map evidence
promote scene meaning
```

Output fields:

```text
source_id
scene_context_candidate
roadside_candidate
indoor_outdoor_candidate
sky_present
ground_present
building_mass_candidate
natural_mass_candidate
map_context_if_available
unknowns
```

### 5. Glyph Region Worker

Job: find text-like visual regions and stabilize them across frames.

Reads:

```text
visual logs
frame samples
region candidates
```

Writes:

```text
glyph_region_candidates.jsonl
```

Does not:

```text
decide final text truth
track personal identity
promote semantic meaning
```

Output fields:

```text
source_id
region_id
frame_range
bounds
glyph_like_score
persistence_frames
motion_stability
candidate_text_if_readable
needs_review
```

### 6. External Label Support Worker

Job: use captions, metadata, search, or operator labels as weak label support.

Reads:

```text
glyph candidates
source metadata
captions
operator labels
approved search results
```

Writes:

```text
label_support_packets.jsonl
```

Does not:

```text
write evidence
promote truth
override visual logs
```

Output fields:

```text
candidate_id
label_source
label_text
source_count
independent_source_count
shape_match_required
glyph_match_required
support_status
```

Law:

```text
Search teaches labels.
Logs remain evidence.
```

### 7. Scene Plate Registry Worker

Job: turn usable images into scene plates for later animation or review.

Reads:

```text
source_manifest.jsonl
plate_bounds.jsonl
geometry_shape_profile.jsonl
```

Writes:

```text
scene_plate_registry.jsonl
```

Does not:

```text
animate
render
invent closeups
```

Output fields:

```text
plate_id
source_id
scene_type
usable_for
camera_allowed
camera_forbidden
state_layers_available
closeup_allowed
safe_crop_bounds
```

### 8. Storyboard State Worker

Job: assign scene plates to scene-state purposes.

Reads:

```text
lyrics
treatment
storyboard contract
scene_plate_registry.jsonl
```

Writes:

```text
storyboard_states.jsonl
```

Does not:

```text
render
animate
override operator treatment
```

Output fields:

```text
scene_state_id
scene_function
plate_id
lyric_range
emotional_state
required_visual_read
allowed_effects
forbidden_effects
state_change_goal
```

### 9. State Motion Worker

Job: define what moves inside a plate.

Reads:

```text
storyboard_states.jsonl
geometry_shape_profile.jsonl
available effect profiles
```

Writes:

```text
state_motion_plan.jsonl
```

Does not:

```text
zoom by default
crop into arbitrary details
hide the subject with effects
```

Default camera rule:

```text
camera_mode = LOCKED_PLATE or MICRO_DRIFT
```

Output fields:

```text
scene_state_id
motion_layers
fog_motion
light_motion
shadow_motion
glyph_motion
ember_motion
parallax_amount
camera_mode
camera_limit_percent
```

### 10. Render Prep Worker

Job: prepare render instructions from scene states.

Reads:

```text
storyboard_states.jsonl
state_motion_plan.jsonl
effect availability records
```

Writes:

```text
render_job_plan.json
```

Does not:

```text
render automatically
approve itself
promote output
```

Output fields:

```text
render_job_id
scene_order
inputs
effects_needed
effects_available
missing_effects
estimated_duration
operator_approval_required
```

### 11. Render Worker

Job: render only approved scenes.

Reads:

```text
render_job_plan.json
operator approval receipt
```

Writes:

```text
generated media
render_manifest.json
render_receipt.json
```

Does not:

```text
promote media as evidence
write source truth
change upstream profiles
```

Output fields:

```text
artifact_id
artifact_kind = generated_media
evidence_status = synthetic_not_evidence
scene_state_id
render_settings
output_path
receipt_id
```

### 12. QA / Review Worker

Job: judge whether output matched the contract.

Reads:

```text
generated media
render_manifest.json
storyboard_states.jsonl
state_motion_plan.jsonl
```

Writes:

```text
review_packet.json
```

Does not:

```text
auto-approve unless explicitly allowed
rewrite source contracts
hide missing pieces
```

Checks:

```text
subject_readable
scene_function_preserved
no_forbidden_crop
no_random_zoom
motion_came_from_state_layers
artifact_marked_synthetic
missing_pieces_listed
```

### 13. Cleanup Worker

Job: remove temporary junk after receipts exist.

Reads:

```text
manifests
receipts
declared temp folders
```

Writes:

```text
cleanup_receipt.json
```

Does not:

```text
delete source files unless explicitly operator-approved
delete durable profiles
delete receipts
clean outside declared roots
```

Checks:

```text
render_manifest_exists
review_packet_exists
artifact_path_recorded
temp_files_identified
operator_approval_if_deleting_raw
```

## Shape / Glyph / Context Consensus

Recognition starts with structure, not truth.

For road signs:

```text
shape finds the candidate
glyphs name the candidate
context checks the candidate
search teaches the label
logs remain the evidence
review promotes the memory
```

Promotion requires:

```text
shape agreement
glyph agreement
context agreement
external label support
repeated local evidence or operator review
```

Three matching external answers are useful only when they are independent
enough to count as separate support. Search output is support, not evidence.

## Compact Memory

Durable memory should be tiny:

```text
signature packs
meter summaries
glyph candidates
region bounds
context tags
receipts
review decisions
```

Source videos and bulky teacher captures are not durable memory unless an
operator explicitly marks them for retention.

## Example: Apple Signature

The system should not store:

```text
I know apple.
```

It should store:

```text
round_or_asymmetric_fruit_body
red_green_yellow_surface_ranges
stem_pocket_geometry
surface_gloss
speckle_texture
hand_table_tree_context_variants
shadow_and_highlight_behavior
motion_persistence
review_status
```

The recognition worker can later compare a new logged region against this
signature pack without claiming truth on its own.

## Example: Stop Sign Candidate

```text
candidate_stop_sign:
  red octagon
  white border
  STOP-like glyph
  roadside/intersection context
  driver-facing angle
  persistence across frames
  rejection_reasons_if_weak
  review_status
```

The candidate becomes a promoted memory only after review or repeated strong
evidence.

## System Law

```text
Do not build a giant app.
Do not build a player.
Do not let one worker become a god bridge.
Do not let generated media enter evidence.
Do not let captions/search become truth.
Do not let camera motion replace state motion.
Do not let a worker promote its own output.
```
