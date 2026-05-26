Made the HTML mockup as a static **Cortex Intake Console** surface.

I counted the UI as:

```text id="kacy7q"
11 major zones
19 panel/card groups
6 assigned tools
5 operator state steps
3 queued sources
6 cleanup operations
5 main truth meters
4 beta alignment cards
```

What each major zone does:

```text id="1zoa4h"
Topbar:
identity, system health, receipt integrity, operator ID, Start/Stop.

Operator State Rail:
single source of truth for PREPARED → ARMED → RECORDING → METERING → REVIEW.

Assigned Tools:
tool cards for Playlist Intake, Virtual Screen, Meter Grid, Angular Cortex, Seismic Cortex, State Focus Lens.

Tools Overview:
compact assigned/active/beta/error count plus tool health.

Virtual Screen Peek-In:
live visual source preview plus source-local truthful meters.

Source Strip:
source filename, duration, codec, profile, signal.

Review Menu:
tool-specific review, beta options, run stats, and receipt access.

Source Queue:
current source, queued sources, progress, estimated time.

Cleanup Operations:
the post-tool cleanup checklist.

Surface Control Law:
hard operator rules so the UI does not lie.

Truth Metrics Overview:
global run truth: resolve, confidence, consistency, coverage, drift, receipts.
```

I did **not** merge the live meter cards and bottom truth meters. They look similar, but they are not redundant:

```text id="ul6wq0"
Live meter cards = source-local truth right now.
Bottom truth meters = run/global truth summary.
```

The redundancy to avoid is scattered capture status. Capture state should come only from:

```text id="fffd9k"
operator-state-rail
virtual-screen-peek
truth-metrics-overview
```

I also baked in bridge/passport hooks for Codex:

```text id="4bn6ap"
data-panel-id
data-tool-id
data-reader-id
data-action-id
data-bridge-id
data-passport-scope
```

So later each panel can get its own bridge, and any inter-panel read/write can require a passported identity. For now it is UI-only, but the hooks are there.
