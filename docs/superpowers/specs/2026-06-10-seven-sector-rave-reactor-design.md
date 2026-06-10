# Seven-Sector Rave Reactor Design

## Decision

Use the second blueprint as the structural source of truth:

- one central vocal reactor
- six surrounding stem sectors
- each sector owns a specific visual behavior
- the center remains the most precise proof element

Use the first blueprint as art direction:

- Cleveland warehouse pressure
- graffiti wall / reactor chamber mood
- steel blue, teal, red, purple, chrome white, pale gold
- haze, strobes, saber-like beams, wet-floor reflection, storm release

## Goal

Create a music-driven TrueVision rave light show proof for:

- `C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup).wav`
- `C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup) Stems (86BPM).zip`

The output should prove that the track operates the visual machine. It should not look like generic background animation.

## Layout

The frame is a seven-part reactor:

```text
                 SYNTH
          saber spin / blade orbit

     DRUMS                   GUITAR
 strobes / impacts       shards / cuts

                 VOCAL
          exact waveform center

      BASS                    KEYS
 tunnel / pressure       glass arcs

                 OTHER
          fog / sparks / atmosphere
```

The central vocal chamber is circular or rounded. The six outer chambers are wedge-like sectors around it. The whole shape should feel like a reactor lens, not a dashboard of boxes.

## Stem Ownership

Every sector has a visible, repeatable job.

- Vocal center: exact waveform, vocal amplitude contour, center pulse, color changes by singer/section.
- Drums: kick impact, snare strobe, hi-hat sparks, fill chases.
- Bass: tunnel expansion, low pressure rings, floor ripple, chamber bend.
- Synth: saber-like rotating beam, blade snap, halo trails.
- Guitar: jagged ribbons, chrome shards, scrape cuts, rough edge bursts.
- Keys: harmonic glass arcs, pale gold/blue linework, prism rings.
- Other: fog, particles, noise sweeps, spray mist, ember drift.

Light may spill between sectors, but control does not. A bass hit may glow into the center, but the viewer should still know the source was bass.

## Timeline Behavior

The renderer should support section-level behavior even before exact lyric timing is added:

- Opening / Singer A: teal vocal waveform, low bass breathing, restrained drums, faint keys, idle synth blade.
- Singer B / tick-tock: red-purple center, drum tick strobes, harder bass tunnel, accelerated synth spin, guitar shards.
- Refrain: hard red vocal lock, snare cracks, bass punches, guitar cuts, sparse cold keys.
- Rebuild: blue-white center, wider keys, softer fog, slower pulse, crown/blade fragments.
- Final chant: white-hot center, full drum strobes, bass max expansion, synth ring, guitar burst, gold key arcs, fog spill.
- Outro: dying haze, low red/blue pulses, slow clock-like decay, blackout.

## Audio Analysis

The renderer should decode the full mix and stems locally. It should derive frame-level envelopes from each stem and map those envelopes to sector state.

Minimum useful signals:

- RMS envelope per stem
- transient/onset proxy per stem
- low/mid/high band energy where practical
- global master energy
- normalized vocal waveform samples for the center

If a named stem is missing, the renderer should degrade gracefully by mapping available stems to the closest sector and recording the fallback in the manifest.

## Output Contract

The first proof should render a short clip before attempting the full song.

Recommended first pass:

- 30 seconds
- 1920x1080 or 1280x720
- 30 fps
- MP4 with the full mix attached
- per-frame or per-section manifest
- receipt listing audio paths, stem mapping, formulas, output path, duration, fps, and limitations

## Boundaries

This is allowed to be a TrueVision-rendered synthetic visual proof. It should not claim to be a trained video foundation model. It should show that audio stems drive a deterministic visual generation system.

No copyrighted Star Wars assets or literal saber branding. The visual language can use spinning blade beams, reactor geometry, hyperspace-like motion, and cinematic haze without copying protected media.

## Acceptance Criteria

- The center visibly follows the vocal waveform.
- The six outer sectors visibly do different jobs.
- Drum hits create hard, readable impacts.
- Bass visibly moves pressure/tunnel geometry.
- Synth creates the cleanest spinning blade behavior.
- Guitar reads as shards or scrape cuts.
- Keys read as harmonic glass arcs.
- Other reads as fog/particles/atmosphere.
- The final output includes a manifest proving what drove what.
