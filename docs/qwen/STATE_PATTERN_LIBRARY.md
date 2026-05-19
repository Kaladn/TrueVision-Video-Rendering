# TrueVision State Pattern Library

The AI does not generate blind. It chooses from known audio/video state patterns and combines them into validated templates.

## Signal Inputs

```text
level
peak_abs
dbfs
delta
peak_event
valley_event
section_energy
```

## Current Patterns

```text
pulse_rings
  peaks trigger expanding rings on a black field

random_geometry_shards
  peaks spawn deterministic random triangles or line shards

quiet_valley_drift
  valleys slow motion and let geometry drift

rising_energy_expansion
  rising level expands geometry and nudges camera push

high_energy_edge_shimmer
  high activity adds edge shimmer and color pressure
```

## Mapping Rule

```text
peaks trigger pulses, flashes, rings, or geometry spawn
valleys slow motion, hold, drift, or dim
rising energy expands geometry and pushes camera
falling energy contracts geometry and cools color
section energy selects scene intensity
```

## Later Fingerprint Libraries

```text
video fingerprints
motion fingerprints
lighting fingerprints
fog/smoke fingerprints
cartoon fingerprints
gameplay fingerprints
music-video fingerprints
camera-language fingerprints
material/surface fingerprints
```

Those future libraries should be learned from real video captures and stored as reusable state-pattern records. They should not claim to recreate the source video unless raw source data is explicitly available and permitted.
