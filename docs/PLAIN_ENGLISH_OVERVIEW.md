# Plain English Overview

TrueVision is a way to describe video as changing visual state.

Instead of saving every full image, the system records compact information about what is happening in small regions of the screen: color, brightness, edges, texture, and motion.

TrueFrameGen uses that saved information to make smoother video. If the recorder captured frame A and then frame B, TrueFrameGen estimates what should happen between them and creates the missing frames.

The important rule is simple:

```text
Recorded state is observation.
Generated frames are reconstruction.
Reconstruction is not proof of what happened.
```

This project is for local video-rendering experiments, music visuals, capture-driven rendering, and frame-smoothing research.
