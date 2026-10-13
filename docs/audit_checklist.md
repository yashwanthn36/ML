# Repository refactoring audit checklist

This audit was performed before the dataset-adapter additions. AVE remains the
only real-data adapter and the only production pretrained-encoder configuration.

- [x] AVE assumptions: `ave_mm.dataset` and AVE split utilities own AVE metadata;
  AVE-specific factories remain adapters. Generic training uses two modality
  tensors, integer labels, IDs, class metadata, and a configured model factory.
- [x] Generic coupling found: report plotting read AVE `label_id` columns and
  forced two protocol names; changed it to use the configured manifest, mapping,
  dataset identity, and arbitrary protocol name.
- [x] Data contract: validate class mapping, labels, IDs, protocol and partition
  overlap before fit/evaluation; optional source-group checks are protocol-driven.
- [x] Media pipeline: generic WAV decoding handles PCM widths and resampling;
  video decoding is shared with AVE. Non-PCM/compressed audio and non-video data
  need a dedicated adapter.
- [x] Split leakage: preserve explicit assignments; added deterministic source
  grouping or sample-stratified split generation that refuses populated split
  columns and reports counts/limitations.
- [x] Provenance: record dataset identity, manifest digest, class mapping, config,
  IDs, checkpoint digests, seed, protocol, package/device details. Generic runs
  reject downstream artifacts from a different manifest.
- [x] Reproducibility: seed Python/NumPy/PyTorch/CUDA and DataLoader workers.
  Deterministic flags are enabled for cuDNN; bitwise equivalence across GPU,
  driver, and library versions is not promised.
- [x] Outputs/dependencies: add package metadata, data docs and ignore rules;
  preserve AVE configs and migrate the misspelled research-reference directory
  to `references/` without changing its files.
- [ ] Production PANNs/ResNet pretrained training is not re-run by this refactor;
  complete AVE experiments remain unverified.
