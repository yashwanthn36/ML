# Reproducibility

Run commands from any directory through the script path only if config and
manifest paths are repository-relative; maintained scripts resolve configs
from the repository root. Prefer launching from the repository root for shell
clarity. Every run records its seed, protocol, config, manifest digest, class
map, sample IDs, encoder configuration and device. Python, NumPy, PyTorch,
CUDA, and DataLoader workers receive deterministic seeds. cuDNN deterministic
mode is enabled; other kernels and different hardware/software stacks can still
prevent bitwise-identical weights. No claim of cross-platform bitwise
reproducibility is made.

Automated tests use tiny local PCM WAV/video fixtures and lightweight CNN
encoders; they validate the pipeline wiring, not production pretrained encoder
accuracy or AVE performance. AVE manifest integrity checks validate its source
metadata, while production model training still requires the AVE media files and
pretrained weights. Training outputs and real datasets are ignored by Git.
