# Experiment protocol

Reference training uses the configured classification loss and validation
macro-F1 checkpointing. Contribution signatures, profile scaling/clustering,
UCD targets, and the reference FCD estimate use the training partition only.
CI-only and proposed CI + MI initialize from the same reference checkpoint and
use the same training/validation data and budget. CI-only has no MI estimator or
alignment gradient. Proposed lambda selection uses validation macro-F1 only.
Test is loaded only for selected-checkpoint evaluation.

Official AVE assignments intentionally retain their known source-video overlap.
AVE `source_video_disjoint` is a separately prepared protocol and must not be
mixed with official reports. Other datasets use their declared protocol name;
the generic adapter does not reinterpret their assignments.

Minority classes use training counts only: all classes at or below the
configured empirical count quantile are selected, with ties included. Test
metrics and plots report missing-positive AP as unavailable. A test metric or
alignment diagnostic is not used for model selection.
