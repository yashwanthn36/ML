# Data layout

Place source datasets under `data/raw/<dataset>/`; generated manifests and split
assignments belong under `data/manifests/<dataset>/`; cached or transformed
features belong under `data/processed/<dataset>/`. These folders are ignored by
Git except this documentation and `.gitkeep` placeholders. Keep media files,
archives, generated caches, and annotations outside source code.

The AVE archive layout and extraction notes are documented in the repository
README. Generic adapter manifests and media paths are documented in
[`docs/dataset_adapter_guide.md`](../docs/dataset_adapter_guide.md).
