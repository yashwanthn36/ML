# Contribution-profile-aware multimodal learning

This repository contains a reusable two-modality learning pipeline and an AVE adapter. It implements contribution signatures, contribution-profile grouping, risk-based UCD targets, Utopia-style mutual-information FCD estimates, class/group-balanced losses, and fixed-weight or minimum-norm two-objective training. The architecture reference is `implementatation/contribution_profile_multimodal_architecture_redesigned.pdf`; the FCD equations are checked against `implementatation/1-s2.0-S1566253525004567-main.pdf`.

## Setup

The project uses Python 3.13. From PowerShell at the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, run `.venv\Scripts\python.exe` without activating. `requirements.txt` includes PyTorch, torchvision, torchaudio, PANNs inference, OpenCV, scikit-learn, PyYAML, imageio-ffmpeg, and pytest.

The AVE media and archive are excluded from Git because they require several gigabytes. Place the archive at `data/raw/AVE/ave.zip` and extracted tracks at `data/raw/AVE/ave/audio/` and `data/raw/AVE/ave/video/`. The manifest and split metadata are tracked. Rebuild manifests from archive metadata with:

```powershell
.\.venv\Scripts\python.exe scripts/data/prepare_ave_manifest.py
.\.venv\Scripts\python.exe scripts/data/prepare_ave_manifest.py --include-source-video-disjoint --seed 17
.\.venv\Scripts\python.exe scripts/data/extract_ave_audio.py
```

The disjoint command writes `ave_manifest_source_video_disjoint.csv` and `ave_source_video_disjoint_summary.json`. It reads ZIP metadata, not the media payloads.

## Dataset and split protocols

`official` preserves the published AVE train/validation/test assignments, including their observed overlap in source-video IDs. Use it as a named benchmark baseline; do not describe its test result as source-video-independent.

`source_video_disjoint` uses deterministic nested `StratifiedGroupKFold` (seed 17 by default). Segments sharing a video ID always stay in one partition. Its summary reports class counts, missing classes, and pairwise source-ID intersections. Source separation takes priority when a class cannot be represented in every partition. Select the protocol in both `dataset.split_protocol` and `test_dataset.split_protocol` in the YAML config. The AVE reader also accepts `temporal_window_policy: full_clip` or `annotated_event` and `min_audio_coverage`.

The configured AVE policy is `full_clip`: audio and video are sampled over the available clip, with 16 kHz mono 16-bit PCM, fixed waveform length, RGB frames, and encoder-side image normalization. The `annotated_event` option synchronously crops both streams to the annotation interval. Short/empty intervals fail with a descriptive error when valid audio would fall below the configured coverage threshold; padding cannot silently dominate. The AVE adapter supplies tensor inputs, while PANNs and ResNet adapters perform their own required preprocessing.

## Model and encoders

`TwoModalityFusion` is dataset-independent: both encoders return token sequences, trainable projections align feature widths, bidirectional cross-attention contextualizes them, a learned sigmoid gate fuses them, and one classifier produces logits. The gate is not an uncertainty estimator. Modality dropout trains learned missing-token paths. A missing token can still attend to the available modality through cross-attention, so the ablations represent the trained missing-modality path rather than deleting an encoder. Changing an absent raw input cannot change its masked representation or the resulting prediction.

The AVE config is the reproducible PANNs CNN14 + ImageNet ResNet-18 baseline. The backbones are frozen; projections, attention, gate, and classifier remain trainable. PANNs handles its audio sample-rate requirements; ResNet handles ImageNet normalization. BEATs and VideoMAE V2 ViT-Giant are not configured or claimed. Encoders remain configurable through model factories/adapters; weights are not downloaded as part of tests.

## Contribution and optimization method

For modality 1 (audio) and modality 2 (video), the saved signed signatures are:

```text
delta_1 = loss_without_1 - full_loss   # mask [0, 1], modality 2 remains
delta_2 = loss_without_2 - full_loss   # mask [1, 0], modality 1 remains
```

Negative deltas are retained. They are used as two-dimensional within-class profile features. Ward clustering standardizes with a scaler fit only on the designated signature-fitting subset, checks group support, and uses a single-group fallback when support/dispersion is insufficient. Centroids and support statistics are fit on that subset; eligible training samples are assigned to the nearest centroid within their class. Missing assignments and label mismatches raise explicit errors.

The UCD target is risk-based: mean positive signed risk increments are normalized, with an epsilon for finite probabilities. Class and class/group targets fall back to their parent scope when support is low or bootstrap stability exceeds the configured threshold. Uncertainty uses a source-video cluster bootstrap when source IDs exist (sample IDs otherwise). Its output is labelled bootstrap spread/quantiles, not calibrated confidence intervals.

FCD is a separate MI estimate; it is not computed from signed loss signatures. The local Utopia paper defines an InfoNCE lower bound in Eqs. (7)–(9): L2-normalized modality and fused features, a positive paired score, a log candidate-count term, and a log-sum-exp over positive and negative candidates. `InfoNCEFCD.per_sample_information` returns signed lower bounds `[anchors, 2]`. Negatives are other current-batch examples plus a detached FIFO bank of previous training examples; matching sample IDs in the queue are excluded. Consequently batch composition and queue contents affect the estimate; batch size 4 alone is a weak contrastive pool, which is why training uses a configurable queue (1024 by default). The reference estimate uses the entire independent estimation subset as its candidate pool. Validation/test estimates do not update a queue.

The paper normalizes MI estimates to get FCD but does not specify how to turn a negative finite-sample InfoNCE bound into a probability. This implementation applies Softplus to the mean bound within each optimization scope, adds epsilon, and normalizes the two modality values. Softplus is an explicit adaptation, not an exact reproduction of Utopia. The result is a finite nonnegative distribution summing to one, and its gradients flow through current representations; queued negatives are detached. Scope-level distributions are computed separately for the selected global, class, or class/group scope. UCD-to-FCD KL uses target UCD as its first argument.

Class-balanced CE uses full-training inverse class frequencies with an N/B scale; under natural uniform minibatch sampling, its mean is an unbiased estimator of equal class risk. Hierarchical group-balanced CE uses full-training class/group counts with the corresponding N/B scale to estimate equal class weighting followed by equal group weighting within each class. It does not self-normalize weights over the current batch, which would make the effective objective composition dependent. Fallback behavior without full counts averages the scopes present in the batch and is marked as an approximation. MOO computes both objective gradients over the same trainable parameter list, handles unused gradients, combines them with the minimum-norm convex coefficient, then assigns gradients and steps once. Diagnostics log gradient norms and cosine similarity.

## Training-only data protocol

By default the reference trainer holds out 20% of the training partition for contribution estimation, using stratified grouping by source video where available. The reference model is fit on the remaining training data and selected using validation macro-F1. Signature fitting, scaling, profile fitting, UCD targets, and the reference FCD use only the independent estimation subset. The remaining training rows may also receive signatures from the held-out-trained reference model solely to assign them to already-fitted centroids; these in-sample assignments are recorded in `contribution_protocol.json` and are not pooled into signature fitting or target estimation. IDs and split protocol are saved under the reference run. Cross-validation is not implemented; this independent holdout is the explicit compute-saving protocol.

An explicit all-training in-sample fallback is available only if the config sets `contribution_estimation_fraction: 0` and `allow_in_sample_contributions: true`. It is statistically weaker and is recorded as such. Validation/test data are not passed into estimators. The test evaluator runs only after checkpoint selection.

## Run order

From the repository root:

```powershell
# Train reference checkpoint; validation chooses the checkpoint.
.\.venv\Scripts\python.exe scripts/training/train_reference.py --config configs/ave_reference.yaml

# Estimate independent training contributions, profiles, and UCD targets.
.\.venv\Scripts\python.exe scripts/training/estimate_contributions.py --config configs/ave_reference.yaml --checkpoint outputs/ave/reference/seed_17/reference_best.pt

# Train the second stage. Fixed-weight runs save one checkpoint per lambda and select by validation macro-F1.
.\.venv\Scripts\python.exe scripts/training/train_joint.py --config configs/ave_reference.yaml --reference-checkpoint outputs/ave/reference/seed_17/reference_best.pt --targets outputs/ave/reference/seed_17/contributions/ucd_targets.json --groups outputs/ave/reference/seed_17/contributions/profile_groups.json

# Evaluate the selected checkpoint once on test.
.\.venv\Scripts\python.exe scripts/training/evaluate_test.py --config configs/ave_reference.yaml --checkpoint <selected-checkpoint-path> --targets outputs/ave/reference/seed_17/contributions/ucd_targets.json
```

Outputs include per-epoch JSONL logs, the selected checkpoint, `experiment_metadata.json` (encoder/checkpoint aliases, preprocessing config, frozen status, cache configuration and package versions), signature CSVs, `reference_fcd.json`, `profile_groups.json`, `ucd_targets.json`, contribution provenance, validation lambda selection, test predictions, and test metrics. Protocol names are included in data metadata, logs/checkpoints, and evaluation output. Default weight aliases are recorded, but checkpoint files are not hashed. Do not choose settings based on test metrics.

## Verification status

Checks executed in this workspace during the audit:

- `python scripts/data/prepare_ave_manifest.py --include-source-video-disjoint --seed 17` completed and generated the alternative manifest and summary.
- The generated disjoint split contains 2,647 train, 665 validation, and 828 test segments; all 28 classes occur in each partition and all source-ID intersections are empty.
- `.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --tb=short` passed 11 tests, including a synthetic reference/signature/FCD/group/UCD/joint/checkpoint/test-evaluator path. No AVE media or pretrained weights were needed.
- `.venv\Scripts\python.exe -m compileall -q src scripts tests` passed.
- `--help` succeeded for reference training, contribution estimation, joint training, test evaluation, feature caching, and AVE manifest preparation.
- The FCD equations were inspected in the local Utopia paper, especially Eqs. (7)–(11), pages 4–5.

No real AVE training, pretrained-weight download, or performance experiment has been run. Although all classes appear in all three generated disjoint partitions for this seed, other datasets/seeds may have missing classes; consult the generated summary. Current unit tests do not validate full PANNs/ResNet inference or real-media decoder behavior.
