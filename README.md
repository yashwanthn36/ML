# Contribution-profile-aware multimodal learning

This repository implements the dataset-agnostic paired-modality framework specified in `implementatation/contribution_profile_multimodal_architecture_redesigned.pdf`. Dataset adapters and YAML configs select data, encoders, contribution estimates, and experiment settings; shared fusion and optimization code follows the common two-modality contract.

## Project layout

```text
configs/
  ave_reference.yaml                 # AVE adapter + reference-run settings
src/
  multimodal/                         # Dataset-agnostic model and training code
    models/fusion.py                  # Generic two-modality fusion head
    models/encoders.py                # Reusable encoder registry and adapters
    contribution/                     # Signed loss signatures and risk-based UCD
    grouping/                          # Within-class Ward profile grouping
    losses/                             # Class/group CE and target-to-current KL
    optim/                              # Two-objective minimum-norm MOO
    training/reference.py             # Generic reference training/evaluation loop
    training/joint.py                  # Fixed-weight/MOO second-stage loop
    data_adapters/cached_features.py  # Frozen-encoder feature cache dataset
  ave_mm/                             # AVE-specific data and raw-media adapters
    dataset.py                        # Manifest reader and synchronized event crops
    models/encoders.py                # Backward-compatible names for shared encoders
    models/__init__.py                # AVE adapter to the generic model factory
scripts/
  data/                               # AVE-only preparation utilities
  training/cache_features.py          # Generic frozen encoder feature caching
  training/estimate_contributions.py  # Signatures, groups and UCD targets
  training/train_reference.py         # Generic config-driven entry point
  training/train_joint.py             # Generic second-stage entry point
  training/evaluate_test.py           # Final full/ablation test evaluation
data/raw/AVE/
  ave/                                # Working data, labels, splits, manifest
    audio/                            # Mono 16 kHz WAV tracks
    video/                            # Original MP4 files
  ave_extras/                         # Dataset README and extraction report
  ave.zip                             # Original source archive
requirements.txt
README.md
```

`.venv/`, temporary setup files, Python bytecode, and `outputs/` are excluded by `.gitignore`.
The extracted audio/video media and original `ave.zip` are also excluded because together they are about 12.7 GB. The AVE manifest, annotations, split lists, and extraction notes are tracked; place the media at the documented `data/raw/AVE/ave/audio/` and `video/` paths after cloning.

## Dataset preparation so far

The supplied AVE archive was inspected and extracted. It contains 4,097 MP4 videos and official annotations and train/validation/test split lists. Audio was extracted from each MP4 as a full-length mono 16 kHz, 16-bit PCM WAV using FFmpeg. The WAV filename and MP4 filename share the source video ID.

`data/raw/AVE/ave/ave_manifest.csv` has one row per official split segment (4,140 rows). Each row includes the sample ID, video ID, class label and integer label ID, quality, official split, annotated event start/end times, and paths to the paired video and audio. The loader uses the event times to crop both modalities to the same interval. It samples eight video frames and pads the cropped audio to ten seconds by default.

Dataset checks found:

- 28 classes and 4,097 source videos in the archive.
- All 4,094 unique videos referenced by the official split lists have both a video and an extracted audio track.
- There are three annotation rows that do not appear in the official split lists; they are therefore absent from the training manifest.
- Official split lists have source-video ID overlap: 10 IDs between train and validation, 8 between train and test, and none between validation and test. The lists are segment-based, so these overlaps should be addressed or explicitly accounted for before making source-independent evaluation claims.
- The extraction report records 4,097 audio files and zero extraction failures.

To rebuild the split manifest from the archive metadata:

```powershell
python scripts/data/prepare_ave_manifest.py
```

To extract or resume audio extraction (existing non-empty WAVs are skipped):

```powershell
python scripts/data/extract_ave_audio.py
```

## Generic code and AVE adapter boundary

The reusable code does not depend on AVE labels, paths, or audio/video field names:

- `TwoModalityFusion` accepts two encoders that each return token features. Learned projections map their features to the shared dimension before bidirectional cross-attention, gated fusion, and a shared classifier. A two-column modality mask selects learned missing-modality tokens.
- The shared model includes separate trainable projections, bidirectional cross-attention, the PDF's learned sigmoid gate, and one classifier reused for full and modality-removal passes. The gate is learned from attended features; it is not an uncertainty estimator.
- The generic training loop consumes batches with `modality_1`, `modality_2`, and `label` keys. It performs natural-sampling cross-entropy training, modality dropout, validation metrics, and checkpoint selection.
- The generic CLI loads dataset and model factories from YAML. A new paired-modality task should need its own dataset adapter, encoders, and config, while the generic fusion/training code can remain unchanged when their input contracts match.
- `multimodal.models.build_model` composes two selected encoders with the same fusion head. Built-ins include `audio_cnn` (trainable mono waveform baseline), `panns_cnn14` (AudioSet-pretrained waveform audio), `frame_cnn` (trainable video-frame baseline), and `resnet18` (ImageNet-pretrained per-frame video). Each emits token features that the shared model projects into its fusion dimension.
- Encoder choices live in the model config, not in the fusion head. A custom encoder from another package can be supplied through a `module:callable` factory; the callable receives `dim` and configured `kwargs` and must return an `nn.Module` with the same token output contract. This lets future datasets use task-appropriate audio, video, text, or other encoders without changing generic training/fusion code.

AVE-specific pieces stay under `src/ave_mm/`: its manifest parsing and media decoding. The AVE adapter maps audio to `modality_1` and video to `modality_2`.

## Environment setup

The project uses Python 3.13. `requirements.txt` pins the direct packages used by the current code, including PyTorch 2.11.0 with CUDA 12.8, matching torchvision and torchaudio builds, PANNs inference, NumPy, OpenCV, scikit-learn, PyYAML, and imageio-ffmpeg. The [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) lists the matching Windows CUDA 12.8 wheels.

Create and activate the local environment in PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The venv was created and packages installed in the current workspace. To use it without activation, call `.venv\Scripts\python.exe` directly. The environment setup was previously checked for package imports, CUDA availability, and dependency consistency. The implementation described below has not yet been compiled or trained as a complete pipeline.

## Reference and contribution stages

`configs/ave_reference.yaml` selects the AVE dataset/model factories and records the experiment settings. The reference run currently uses:

- seed 17, batch size 4, up to 30 epochs, and early stopping after five epochs without validation macro-F1 improvement;
- AdamW with learning rate `3e-4` and weight decay `1e-4`;
- natural training-split sampling and ordinary cross-entropy;
- modality dropout probability 0.2, so the learned missing-modality path receives training examples;
- eight video frames at 112 by 112 pixels and ten seconds of 16 kHz audio input.

The AVE config selects PANNs CNN14 pretrained on AudioSet for audio and ImageNet-pretrained ResNet-18 for video. Both backbones are frozen, while shared projections, attention, fusion, and classifier parameters are trainable. The PANNs encoder resamples the configured 16 kHz input to its expected 32 kHz rate and returns a pooled audio embedding token; its checkpoint downloads from the official PANNs release on first use. ResNet-18 expects RGB floats in `[0, 1]` and applies ImageNet normalization; AVE frames are resized to 224 by 224. These are configurable defaults, not hard-coded dataset assumptions. Other modalities and pretrained model families can use the encoder registry or a custom factory. A custom encoder should implement its own preprocessing and freezing policy. Validation macro-F1 selects the checkpoint. The test split is not used by the trainer.

The shared fusion model has trainable projections, bidirectional cross-attention, the PDF's learned sigmoid gate, one classifier, and learned missing-modality tokens trained with modality dropout. `model.freeze_encoders` controls whole-encoder freezing in the reference run; ResNet can freeze its ImageNet backbone independently while leaving the shared projection trainable. The second-stage trainer freezes both encoders after loading the reference checkpoint, matching the architecture's trainable-parameter set. For frozen encoders, `cache_features.py` stores per-example token sequences. It caches train, validation, and configured test splits. `multimodal.data_adapters.cached_features:build_datasets` loads training/validation caches, while `build_test_dataset` loads the test cache. Set `cached_features: true` in the model config to replace raw encoders with identities while training projections, fusion, and classifier.

`train_reference.py` reserves a stratified 20% subset of the training split for contribution estimation by default and writes its sample IDs. The reference model is fit on the remaining training samples; validation remains separate. `estimate_contributions.py` measures full/A-removed/V-removed losses on the held-out training subset, saves signed deltas and reference FCD, fits per-class Ward groups with a training-only scaler and silhouette checks, then assigns the remaining training samples to their nearest class centroids. It estimates global/class/group risk-based UCD targets, bootstraps their uncertainty, and applies configured support/instability fallbacks.

`InfoNCEFCD` follows the local Utopia paper's Eqs. (7)–(9): paired projected modality features and the fused representation are L2-normalized, and batchwise cosine scores estimate an InfoNCE mutual-information lower bound. The lower bound can be negative, so the current implementation applies softplus before normalization; this is an explicit adaptation to meet the architecture PDF's nonnegative-distribution requirement, not a transform specified by Utopia. The risk-based UCD, target-first KL, modality-removal signature, and temporal mean pooling are implemented separately. The cited paper's PDF is `implementatation/1-s2.0-S1566253525004567-main.pdf`.

`train_joint.py` supports class-balanced or hierarchical class/group-balanced CE and fixed-weight or the PDF's two-objective minimum-norm convex gradient update. Fixed-weight runs evaluate the configured log-spaced `lambda_mi_grid` using validation macro-F1 and save the selected checkpoint; MOO recomputes the gradient combination each step. Training logs include objective values, gradient norms/cosines, and validation macro-F1, balanced accuracy, accuracy, and per-class recall.

For a raw-media reference run:

```powershell
.\.venv\Scripts\python.exe scripts/training/train_reference.py --config configs/ave_reference.yaml
.\.venv\Scripts\python.exe scripts/training/estimate_contributions.py --config configs/ave_reference.yaml --checkpoint outputs/ave/reference/seed_17/reference_best.pt
```

For joint training, use a config whose `contribution.target_granularity` is `global`, `class`, or `group`; set `training.ci_objective` to `class_balanced` or `group_balanced`, and `training.optimization_mode` to `fixed_weight` or `moo`:

```powershell
.\.venv\Scripts\python.exe scripts/training/train_joint.py --config configs/ave_reference.yaml --reference-checkpoint outputs/ave/reference/seed_17/reference_best.pt --targets outputs/ave/reference/seed_17/contributions/ucd_targets.json --groups outputs/ave/reference/seed_17/contributions/profile_groups.json
```

After validation checkpoint selection, pass the selected checkpoint path printed by `train_joint.py` to the final test evaluator:

```powershell
.\.venv\Scripts\python.exe scripts/training/evaluate_test.py --config configs/ave_reference.yaml --checkpoint <selected-checkpoint-path> --targets outputs/ave/reference/seed_17/contributions/ucd_targets.json
```

The test evaluator saves paired predictions, audio-only/video-only ablations through the trained missing-modality path, per-class metrics, test FCD, and UCD–FCD KL discrepancy. Do not use its results to choose settings.

Run the reference model from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/training/train_reference.py --config configs/ave_reference.yaml
```

The selected checkpoint is written to `outputs/ave/reference/seed_17/reference_best.pt`.

## Validation status

The added PANNs dependencies installed successfully in `.venv`. The latest implementation changes have not been compiled, unit-tested, or trained. No model training or evaluation results are available yet.

## Remaining implementation work

1. Decide how to handle the source-video overlap across official AVE splits; preserve the official protocol as a named baseline and avoid presenting it as source-independent.
2. Compare the selected AudioSet PANNs and ImageNet ResNet encoders against domain-appropriate alternatives for other datasets; encoder choice remains configurable through each dataset/model config.
3. Run controlled multi-seed comparisons across class/group losses, target granularities, fixed-weight/MOO, and modality ablations; evaluate the test split only after validation selection.

