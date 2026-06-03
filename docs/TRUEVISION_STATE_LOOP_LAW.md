# TrueVision State Loop Law

TrueVision is not a normal media recorder or renderer.

Normal media words are compatibility terms only:

```text
record = witness external surface into state
render = surface planned state as media
```

The canonical TrueVision loop is:

```text
Witness -> Profile -> Plan -> Replay -> Surface
```

## Terms

```text
Witness
Observe an external visual/audio/media surface and encode state.

Profile
Extract reusable behavior from state.

Plan
Create a forward or reverse state path from a behavior profile and goal.

Replay
Walk a state path over time.

Surface
Turn state into visible or audible media at the edge.
```

## Law

```text
TrueVision does not record video to make video.
TrueVision witnesses state to learn behavior.
TrueVision does not render effects from copies.
TrueVision surfaces planned state as media.
```

## Two-Way Behavior Families

Every serious TrueVision behavior family should be two-way unless proven otherwise.

That means a behavior family should be able to support this loop:

```text
surface/media example -> Witness -> Profile
Profile + desired scene -> Plan -> Replay -> Surface
Surface -> Witness -> Compare -> Adjust
```

Individual tools may implement only one stage, but the behavior family remains
bidirectional.

Examples:

```text
gravity_collision_decay
  witness bounce state
  profile gravity, velocity, collision, decay, contact shadow
  plan a new bounce path
  replay the state path
  surface it as visible motion

branching_discharge
  witness lightning-like state
  profile luma spike, branch growth, bloom spread, afterglow decay
  plan a new discharge state
  replay the discharge path
  surface it as a new bolt-like event

fog_reveal
  witness haze and object reveal
  profile density, scatter, depth fade, edge recovery, reveal rate
  plan new fog behavior
  replay the fog state
  surface it as atmosphere
```

## Source Truth Boundary

```text
Raw media is optional input or output.
Raw media is not TrueVision source truth.
State is the source truth.
Behavior profiles are reusable memory.
Rendered media is a surface, not evidence.
```

