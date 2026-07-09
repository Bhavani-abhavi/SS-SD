# SS-SD: Kinematic-Conditioned Stable Diffusion for Surgical Suturing

Fine-tunes Stable Diffusion 1.5 to generate realistic surgical video frames conditioned on
robotic motion data.

## What it does
- Replaces the CLIP text encoder with a custom `KinematicEncoder` mapping 76-dimensional
  robot kinematics + gesture labels into the model's cross-attention interface
- Fine-tunes with LoRA for parameter efficiency on the JIGSAWS Suturing dataset
- Evaluated across PSNR, SSIM, FID, and optical flow metrics
- Ships an interactive Streamlit app for real-vs-generated video comparison

## Stack
PyTorch · Stable Diffusion · LoRA · Streamlit
