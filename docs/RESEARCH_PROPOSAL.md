# Future master's research direction

## Research question

**Can a Vehicle–RSU RGB system infer and maintain a camera-derived relative-pose distribution from static scene evidence over time, then make coverage-preserving cross-view association decisions that abstain when predicted merge risk is too high?**

This is a proposal, not a completed camera-only result.

## Rationale and required evidence

The TUMTraf study establishes RSU sensing complementarity but shows that forced geometry-only association can discard valid coverage. Vehicle lifting improves under privileged pose, whereas the tested pose-free object graph and all-class image reprojection fail their gates.

Future work requires: (1) leakage-safe scene-disjoint sequential evaluation; (2) documented image state, intrinsics, extrinsics, and coordinate convention; (3) a strict RGB-only inference firewall, with LiDAR/GNSS/GT offline only; (4) separate reporting of union coverage, accepted-pair precision, false merges, duplicates, abstention, pose error, and coverage loss; and (5) pre-registered acceptance criteria before held-out measurement.

The intended contribution is an evidence-based characterization of when camera-only cooperation should associate, abstain, or retain separate hypotheses—not novelty for Hungarian matching, PnP, or standard detection alone.
