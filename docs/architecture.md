# Architecture and dataset interface

The training, contribution, and evaluation code consumes the canonical paired
sample mapping:

| Key | Type | Meaning |
|---|---|---|
| `modality_1`, `modality_2` | tensors | aligned audio/video inputs; batch dimensions must agree |
| `label` | scalar `torch.long` | integer in `[0, num_classes)` |
| `sample_id` | unique non-empty string | stable across runs and splits |
| `label_name` | string | must match the dataset-wide class map |
| `source_id` | optional string | source recording/group identity |
| `split`, `split_protocol` | strings | assignment and declared experiment protocol |

Each adapter exposes `labels`, `sample_ids`, `class_names`, `class_to_id`,
`num_classes`, `split`, `split_protocol`, optional `source_ids`, and a `metadata`
mapping. Training and test factories return train/validation datasets and the
test dataset respectively. `validate_dataset` and `validate_split_datasets`
check this contract, class mapping stability, duplicate IDs, and partition
separation before a run proceeds. Tensor shapes are encoder-specific; the model
encoder adapters return token sequences `[batch, tokens, feature_dim]`.

The AVE adapter remains responsible for AVE annotation windows and official
split semantics. `GenericAudioVideoDataset` reads configurable manifest columns
and uses the same multimodal trainers unchanged. Use a separate adapter if a
dataset has different annotation semantics, compressed audio, nonstandard
video, or asynchronous modalities.
