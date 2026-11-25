# Reference-pose-assisted architecture

```mermaid
flowchart LR
    V[Vehicle RGB image] --> VD[YOLO11s Vehicle detections]
    VD --> VL[Class-size-prior Vehicle ground localization]
    R[RSU RGB image] --> RD[YOLO11s RSU detections]
    RD --> RL[RSU ground-plane localization]
    P[LiDAR-derived per-frame Vehicle-to-S110 pose\nPrivileged input] --> VL
    VL --> S[Estimated S110 object positions]
    RL --> S
    S --> H[Fixed 4 m class-compatible\n(5.5 m bus/truck) Hungarian]
    H --> O[Fused and unmatched outputs]
```

The frozen class-compatible Hungarian distance gate is 4 m, extended to 5.5 m for bus/truck pairs. The LiDAR-derived pose supports this archived baseline only. The evaluated Hungarian rule causes destructive merges; naïve concatenation is retained as the coverage upper baseline. Future strict camera-only work must replace pose with a validated visual relative-alignment estimate and make coverage-preserving association/abstention decisions.

