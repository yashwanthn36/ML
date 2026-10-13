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

FCD is a separate MI estimate; it is not computed from signed loss signatures. The local Utopia paper defines an InfoNCE lower bound in Eqs. (7)–(9): L2-normalized modality and fused features, a positive paired score, a log candidate-count term, and a log-sum-exp over positive and negative candidates. `InfoNCEFCD.per_sample_information` returns signed lower bounds `[anchors, 2]`. Negatives are other current-batch examples plus a detached FIFO bank of previous training examples; matching sample IDs are excluded, and AVE sets `exclude_same_source_negatives: true` to exclude sibling source-video segments from current and queued negatives. Sample IDs and source IDs are stored in queue entries. Datasets without source IDs fall back to sample-ID exclusion. Insufficient candidate pools raise an actionable error. This configurable negative-sampling policy is an AVE experiment choice, not an exact reproduction of Utopia. Consequently batch composition and queue contents affect the estimate; batch size 4 alone is a weak contrastive pool, which is why training uses a configurable queue (1024 by default). The reference estimate uses the entire independent estimation subset as its candidate pool. Validation/test estimates do not update a queue.

The paper normalizes MI estimates to get FCD but does not specify how to turn a negative finite-sample InfoNCE bound into a probability. This implementation applies Softplus to the mean bound within each optimization scope, adds epsilon, and normalizes the two modality values. Softplus is an explicit adaptation, not an exact reproduction of Utopia. The result is a finite nonnegative distribution summing to one, and its gradients flow through current representations; queued negatives are detached. Scope-level distributions are computed separately for the selected global, class, or class/group scope. UCD-to-FCD KL uses target UCD as its first argument.

Class-balanced CE uses full-training inverse class frequencies with an N/B scale; under natural uniform minibatch sampling, its mean is an unbiased estimator of equal class risk. Hierarchical group-balanced CE uses full-training class/group counts with the corresponding N/B scale to estimate equal class weighting followed by equal group weighting within each class. It does not self-normalize weights over the current batch, which would make the effective objective composition dependent. Fallback behavior without full counts averages the scopes present in the batch and is marked as an approximation. MOO computes both objective gradients over the same trainable parameter list, handles unused gradients, combines them with the minimum-norm convex coefficient, then assigns gradients and steps once. Diagnostics log gradient norms and cosine similarity.

## Training-only data protocol

By default the reference trainer holds out 20% of the training partition for contribution estimation, using stratified grouping by source video where available. The reference model is fit on the remaining training data and selected using validation macro-F1. Signature fitting, scaling, profile fitting, UCD targets, and the reference FCD use only the independent estimation subset. The remaining training rows may also receive signatures from the held-out-trained reference model solely to assign them to already-fitted centroids; these in-sample assignments are recorded in `contribution_protocol.json` and are not pooled into signature fitting or target estimation. IDs and split protocol are saved under the reference run. Cross-validation is not implemented; this independent holdout is the explicit compute-saving protocol.

An explicit all-training in-sample fallback is available only if the config sets `contribution_estimation_fraction: 0` and `allow_in_sample_contributions: true`. It is statistically weaker and is recorded as such. Validation/test data are not passed into estimators. The test evaluator runs only after checkpoint selection.

## AVE protocol configurations and run order

Use `configs/ave_official.yaml` for the published AVE assignments and `configs/ave_source_video_disjoint.yaml` for the deterministic source-video-disjoint split (seed 17). Each declares the same protocol in `dataset.split_protocol` and `test_dataset.split_protocol`, and uses a separate root: `outputs/ave/official/` or `outputs/ave/source_video_disjoint/`. Seeded artifacts live below `seed_17/`. `configs/ave_reference.yaml` remains a backward-compatible alias of the official setup and uses the official output root.

Run one protocol at a time from the repository root. Do not mix artifact paths between the two command groups. Provenance validation checks protocol, seed, configuration, sample IDs, and checkpoint SHA-256; output directories carry a protocol marker and reject incompatible reuse. Older artifacts without provenance require the explicit `--allow-legacy-artifacts` option on contribution estimation, joint training, or evaluation. Legacy output directories without a marker must be moved or archived before reuse.

Official protocol:

```powershell
$cfg = "configs/ave_official.yaml"
$root = "outputs/ave/official/seed_17"
.\.venv\Scripts\python.exe scripts/training/train_reference.py --config $cfg
.\.venv\Scripts\python.exe scripts/training/estimate_contributions.py --config $cfg --checkpoint "$root/reference_best.pt"
.\.venv\Scripts\python.exe scripts/training/train_joint.py --config $cfg --reference-checkpoint "$root/reference_best.pt" --targets "$root/contributions/ucd_targets.json" --groups "$root/contributions/profile_groups.json"
.\.venv\Scripts\python.exe scripts/training/evaluate_test.py --config $cfg --checkpoint "$root/joint/lambda_0.1/joint_best.pt" --targets "$root/contributions/ucd_targets.json"
```

Source-video-disjoint protocol:

```powershell
$cfg = "configs/ave_source_video_disjoint.yaml"
$root = "outputs/ave/source_video_disjoint/seed_17"
.\.venv\Scripts\python.exe scripts/training/train_reference.py --config $cfg
.\.venv\Scripts\python.exe scripts/training/estimate_contributions.py --config $cfg --checkpoint "$root/reference_best.pt"
.\.venv\Scripts\python.exe scripts/training/train_joint.py --config $cfg --reference-checkpoint "$root/reference_best.pt" --targets "$root/contributions/ucd_targets.json" --groups "$root/contributions/profile_groups.json"
.\.venv\Scripts\python.exe scripts/training/evaluate_test.py --config $cfg --checkpoint "$root/joint/lambda_0.1/joint_best.pt" --targets "$root/contributions/ucd_targets.json"
```

`lambda_0.1` is an example selected checkpoint path; use the path printed by `train_joint.py` after validation selection. Do not select hyperparameters from test results. Each stage records protocol/provenance in its output, logs, checkpoints, and final metrics. Contribution targets and profile assignments are tied to the exact reference checkpoint hash and training-ID hash.

## AVE annotation integrity

Run `python scripts/data/prepare_ave_manifest.py --include-source-video-disjoint --seed 17` to regenerate the manifests and reproducible JSON integrity reports. The audit reports exact annotation duplicates, repeated source/label/interval combinations, source-level label conflicts, overlapping or identical event intervals, missing/invalid intervals, source-ID split overlap, and manifest/summary consistency, including sample IDs and annotation fields. Original rows are retained in both manifests; exact duplicates are reported and remain eligible for training. Distinct non-overlapping segments from a source video are not deduplicated. The disjoint split assigns all segments from one source to exactly one partition. Conflicts are reported rather than silently relabelled.

The AVE configs set `contribution.fcd_kwargs.exclude_same_source_negatives: true`. Other current-batch and queued segments from the same source video are excluded as negatives while the positive paired example is retained. Datasets without source IDs fall back to sample-ID exclusion; too few valid candidates produce an explicit error. This policy is part of the experiment design and is not claimed to reproduce Utopia exactly. The FCD probability transform also uses Softplus because finite-sample InfoNCE bounds may be negative; it is an adaptation, not an exact reproduction.

## Device selection

Run commands accept `--device auto|cuda|cpu`; the config may also set `device`. `auto` selects CUDA when PyTorch reports it available, otherwise CPU. Startup logs show the selected device, GPU name, CUDA availability, and PyTorch CUDA version; CPU runs explicitly report CPU. An explicit `--device cuda` fails if CUDA is unavailable. Model and batch placement is checked, and the device is recorded in metadata and checkpoints. Check the local installation with `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`. Device selection does not measure GPU speed.

## Verification status

The reliability suite exercises tiny reference-to-joint training paths for both protocols, compatible and deliberately mismatched artifacts, manifest audit cases, FCD source/sample exclusions, and CPU/CUDA selection behavior. Run these checks from the repository root:

```powershell
python -m pytest -q -p no:cacheprovider tests
python -m compileall -q src scripts tests
python scripts/training/train_reference.py --help
python scripts/training/estimate_contributions.py --help
python scripts/training/train_joint.py --help
python scripts/training/evaluate_test.py --help
python scripts/training/cache_features.py --help
python scripts/data/prepare_ave_manifest.py --help
```

Results for this reliability pass: the pytest command passed all 23 tests; `compileall` succeeded; all six CLI help commands exited successfully. The manifest command `python scripts/data/prepare_ave_manifest.py --include-source-video-disjoint --seed 17` regenerated both reports with consistent summaries, no exact duplicate rows, 45 cross-label source conflicts, 47 overlapping/identical interval pairs, and one invalid interval. A direct AVEDataset read/decode of the first training sample succeeded under both protocols using the extracted local audio/video. A small fusion forward/backward smoke also ran on the detected NVIDIA GeForce RTX 3050 6GB Laptop GPU (PyTorch CUDA 12.8); this verifies basic CUDA placement and execution, not full AVE training or speed.

No full pretrained-model training or performance experiment has been run. The official protocol retains published source-video overlap; use the disjoint protocol when source-level independence is required. The invalid interval and annotation label conflicts remain reported in the integrity JSON and are not automatically repaired or removed.
