Recommended approach
Component	Format	Why
Dataset exploration and visualization	.ipynb	Inspect samples, class distributions, and modality alignment
Feature extraction and caching	.py	Reusable, consistent processing
Model architecture and fusion	.py	Keep the model organized
Contribution estimation (UCD/FCD)	.py	Reusable mathematical implementation
Within-class contribution grouping	.py	Reproducible clustering
CI and MI loss functions	.py	Test and validate each objective independently
Fixed-weight and MOO optimization	.py	Run controlled experiments
Evaluation and result analysis	Both	.py for metrics; .ipynb for plots and comparisons
Suggested project structure
multimodal-longtail/

configs/

Dataset and training settings

data/

Loading and preprocessing

models/

Encoders and fusion

contributions.py

Ablation-based signatures

grouping.py

Within-class clustering

losses.py

CI and MI objectives

train.py

Training and optimization

evaluate.py

Metrics and ablations

notebooks/

01_dataset_analysis.ipynb
02_contribution_analysis.ipynb
03_results_visualization.ipynb

results/

Metrics, plots, checkpoints



How we should implement it
1. Start in a notebook to load one dataset, verify audio/video alignment, inspect class counts, and test feature extraction.
2. Move reusable code into .py files as soon as each component works.
3. Implement and validate the contribution signatures and UCD/FCD calculations separately before combining them with the CI objective.
4. Run experiments through train.py using configuration files and fixed random seeds.
5. Use notebooks to compare results across global, class-wise, and group-wise targets, and fixed-weight versus MOO optimization.
My recommendation: Since this is a research project requiring controlled experiments, reproducibility, and comparisons across multiple datasets, don't implement everything in one large notebook. Start with one notebook for exploration, then build the actual pipeline using modular .py files.