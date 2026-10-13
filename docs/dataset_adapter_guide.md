# Adding a paired audio-video dataset

## Generic manifest adapter

The generic adapter expects a CSV with one row per paired sample and these
configurable logical fields: `sample_id`, `audio_path`, `video_path`, `label`,
and `split`; `source_id` is optional. If source IDs are present they identify
recordings, not segments. IDs must be unique across the full manifest. All
partitions must use the same class map. An explicit map is recommended for
long-lived datasets; otherwise class names are sorted once across the complete
manifest. The same map is used by the train, validation, and test factories.

The generic reader supports uncompressed PCM WAV (8, 16, 24, or 32 bit), mono
conversion, linear resampling, fixed-length zero padding/truncation, and
OpenCV-decodable video containers (commonly MP4/AVI/MOV/MKV). Videos are sampled
evenly over available duration into RGB float tensors `[T,C,H,W]`. Relative
media paths resolve from `media_root` by default; set
`media_paths_relative_to: manifest` to resolve paths from the CSV directory.
Relative manifest and media-root config paths resolve from the repository root.
Unsupported codec or decode failures are errors; media are not silently
dropped. This is not a universal decoder for arbitrary media.

Copy [`configs/datasets/dataset_template.yaml`](../configs/datasets/dataset_template.yaml)
and edit its dataset name, manifest/media paths, mapping, split protocol and
preprocessing. Its PANNs/ResNet settings are production pretrained choices that
may download weights. The CPU synthetic integration test uses local lightweight
CNN encoders and does not download weights.

Example generic adapter block:

```yaml
dataset:
  name: my_dataset
  adapter: generic_audio_video
  factory: multimodal.data_adapters.generic_audio_video:build_datasets
  split_protocol: official
  kwargs:
    manifest_path: data/manifests/my_dataset/paired_av.csv
    media_root: data/raw/my_dataset
    media_paths_relative_to: media_root
    columns: {sample_id: sample_id, audio_path: audio_path, video_path: video_path,
              label: category, source_id: recording_id, split: partition}
    class_mapping: {class_a: 0, class_b: 1, class_c: 2}
    num_frames: 8
    image_size: 224
    sample_rate: 16000
    audio_samples: 160000
test_dataset:
  name: my_dataset
  adapter: generic_audio_video
  factory: multimodal.data_adapters.generic_audio_video:build_test_dataset
  split_protocol: official
  kwargs: # same manifest, columns, mapping, media root and preprocessing
```

The actual template contains a complete model, contribution, training, and
evaluation config. Keep `dataset` and `test_dataset` manifests, mappings,
preprocessing, protocol, and identities synchronized.

## Splits and validation

Keep published train/validation/test assignments as-is and give them a dataset
specific protocol name. Never feed official splits to the split generator. For
an unsplit CSV, create a *new* output manifest:

```powershell
python scripts/data/split_manifest.py --input data/manifests/my_dataset/unsplit.csv --output data/manifests/my_dataset/split_seed_17.csv --seed 17
```

When source IDs are available this command uses grouped splitting and will fail
if they are absent. Grouping prevents source leakage but class stratification is
best-effort; inspect the generated `.split_summary.json` for per-class support
and partition sizes. To permit stratified sample-level splitting when source
IDs are missing, explicitly pass `--allow-sample-level`. This can leak source
recordings across partitions. Existing non-empty split columns are never
overwritten. The generic adapter itself only consumes explicit assignments.

Run all three conditions using the configured reference, CI-only, and proposed
CI + MI commands in the README. For a new adapter, the trainers, contribution
estimator, evaluator, and report script are reused unchanged. Implement a new
adapter only for incompatible media or annotation semantics. Verify official
class maps, source policy, corrupt media behavior, and preprocessing before
interpreting results.

## Artifacts and provenance

Results are under `outputs/<dataset>/<protocol>/seed_<seed>/`: reference
checkpoints/logs, contribution signatures/groups/UCD/FCD, CI-only and proposed
checkpoints/logs, per-condition evaluations, then protocol-separated reports.
The provenance records config, manifest hash, class mapping, sample-ID hashes,
protocol, seed, encoders, dependency/device details and reference-checkpoint
identity. Never compare results with different manifest hashes or class maps.
