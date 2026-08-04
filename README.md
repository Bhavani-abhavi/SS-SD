# SS-SD: Kinematic-Conditioned Stable Diffusion for Surgical Suturing

Fine-tunes Stable Diffusion 1.5 to generate realistic surgical video frames conditioned
directly on robotic motion data, rather than text prompts.

## Problem

Surgical robotics research needs large volumes of realistic training video, but real
surgical footage is scarce, expensive to label, and privacy-constrained. This project
explores whether a generative video model can be conditioned on robot kinematics to
synthesize physically plausible surgical footage for data augmentation, simulation, and
downstream perception-model training.

## Approach

- Replaced Stable Diffusion 1.5's CLIP text encoder with a custom `KinematicEncoder` that
  projects 76-dimensional robot kinematics + gesture labels into the UNet's cross-attention
  conditioning space, in place of text embeddings.
- Fine-tuned with LoRA (via `peft`) for parameter-efficient adaptation instead of full
  fine-tuning.
- Trained and evaluated on the JIGSAWS Suturing dataset (real surgical robot kinematics +
  video).
- Added a YOLO-based detection pipeline for automated labeling of surgical tool/gesture
  events, used as auxiliary supervision.

## Repository structure

- `notebooks/` — data ingestion & alignment, detection/labeling, kinematic analysis,
  ControlNet synthesis + dashboard, and a Colab training notebook
- `scripts/` — training (`train_sd.py`, `train_yolo.py`), inference (`inference_sd.py`,
  `run_synthesis.py`, `run_detection.py`), evaluation (`metrics_on_grid.py`,
  `video_quality_metrics.py`, `generate_eval_grid.py`, `generate_eval_video.py`), data prep
  (`download_jigsaws.py`, `prepare_trials.py`, `prepare_labeling_set.py`,
  `compute_kinematics.py`), and a Streamlit comparison app (`streamlit_compare.py`,
  `launch_dashboard.py`)
- `src/suturing_pipeline/` — core pipeline package (KinematicEncoder, dataset loaders,
  model wiring)
- `configs/` — training and inference configuration files
- `tests/` — unit tests
- `outputs/`, `versions/` — generated artifacts and checkpoints

## Evaluation

Model outputs are benchmarked against real JIGSAWS footage with PSNR, SSIM, FID, and
optical-flow consistency metrics, plus an interactive Streamlit app
(`scripts/streamlit_compare.py`) for side-by-side real-vs-generated video review.

## Stack

PyTorch · Diffusers · PEFT (LoRA) · Transformers · Ultralytics YOLO · OpenCV · Streamlit

## Setup

```
pip install -r requirements.txt
python scripts/download_jigsaws.py
```

See `scripts/` for individual training, inference, and evaluation entry points, and
`configs/` for their corresponding configuration files.
