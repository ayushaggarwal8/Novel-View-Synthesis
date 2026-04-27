# Novel View Synthesis — Pipeline Explained

This document describes how both pipelines in this repo work end to end: what data flows through each stage, what is actually being optimized during training, and what the loss functions compute.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Shared Preprocessing](#2-shared-preprocessing)
   - 2.1 [Frame Extraction](#21-frame-extraction)
   - 2.2 [COLMAP — Structure from Motion](#22-colmap--structure-from-motion)
3. [Pipeline A — 3D Gaussian Splatting](#3-pipeline-a--3d-gaussian-splatting)
   - 3.1 [What is a 3D Gaussian?](#31-what-is-a-3d-gaussian)
   - 3.2 [Scene Representation](#32-scene-representation)
   - 3.3 [Differentiable Rasterization](#33-differentiable-rasterization)
   - 3.4 [What is Being Trained?](#34-what-is-being-trained)
   - 3.5 [Loss Function](#35-loss-function)
   - 3.6 [Densification and Pruning](#36-densification-and-pruning)
   - 3.7 [Novel View Rendering](#37-novel-view-rendering)
4. [Pipeline B — Diffusion-Based NVS](#4-pipeline-b--diffusion-based-nvs)
   - 4.1 [Zero123++ Overview](#41-zero123-overview)
   - 4.2 [How Diffusion Models Work](#42-how-diffusion-models-works)
   - 4.3 [How Zero123++ Conditions on Viewpoint](#43-how-zero123-conditions-on-viewpoint)
   - 4.4 [Inference at Runtime](#44-inference-at-runtime)
5. [Comparison of Approaches](#5-comparison-of-approaches)
6. [Glossary](#6-glossary)

---

## 1. Problem Statement

Given a monocular video of a scene (a robot in an environment), synthesize photorealistic images from camera viewpoints that were never observed in the original footage.

This is the **Novel View Synthesis (NVS)** problem. The core challenge is that a 2D video gives only partial information about 3D geometry — the system must infer what occluded surfaces look like and how lighting behaves from unseen angles.

---

## 2. Shared Preprocessing

### 2.1 Frame Extraction

**Script:** `gaussian_splatting/scripts/01_extract_frames.py`

The input video is decoded with OpenCV (`cv2.VideoCapture`). Rather than taking every frame (which would create thousands of nearly identical images), we sample at **0.5 fps** — one frame every 2 seconds. For a 4-minute video this gives roughly 120 candidate frames.

A **redundancy filter** then discards frames that are too similar to the previously kept frame (mean absolute pixel difference below a threshold of 5/255). This avoids feeding COLMAP pairs of frames that are essentially identical, which wastes compute and can destabilize the reconstruction.

Frames are resized so the longer edge is at most 1600px (COLMAP's practical limit for SIFT extraction speed) and saved as sequentially numbered JPEGs: `frame_00001.jpg`, `frame_00002.jpg`, ...  The sequential numbering is important — COLMAP's sequential matcher uses it to know which frames are temporally adjacent.

### 2.2 COLMAP — Structure from Motion

**Script:** `gaussian_splatting/scripts/02_run_colmap.py`

COLMAP solves the **Structure from Motion (SfM)** problem: given a set of images of a scene taken from unknown positions, simultaneously recover (a) the 3D positions of scene points and (b) the 6-DoF camera pose for each image.

The pipeline runs four stages:

**Stage 1 — Feature Extraction**
COLMAP runs SIFT (Scale-Invariant Feature Transform) on every image, detecting keypoints (distinctive corners/blobs) and computing 128-dimensional descriptors for each one. Keypoints are localized at multiple scales so they are invariant to zoom.

**Stage 2 — Feature Matching**
For a video with temporally ordered frames, we use **sequential matching**: each frame is matched only against its N nearest temporal neighbors (overlap=10). This is critical for moving-camera video — exhaustive matching (all-pairs) finds spurious correspondences between far-apart frames where the scene geometry is inconsistent, causing COLMAP's mapper to fail or produce a fragmented reconstruction.

**Stage 3 — Sparse Reconstruction (Mapper)**
COLMAP runs incremental Structure from Motion:
- Selects a good initial pair of frames with sufficient baseline and overlap
- Triangulates 3D points from matched 2D correspondences via the epipolar constraint
- Registers additional cameras one at a time using PnP (Perspective-n-Point) — given known 3D points and their 2D projections, solve for the camera pose
- Runs bundle adjustment periodically to jointly refine all camera poses and 3D point positions by minimizing reprojection error

The output is a **sparse point cloud** (typically a few thousand points for an 80-frame sequence) and a precise camera pose (rotation + translation) for each registered image.

**Stage 4 — Image Undistortion**
The COLMAP reconstruction uses a SIMPLE_RADIAL camera model (single focal length + one radial distortion coefficient). The image undistorter warps all images to be consistent with a perfect PINHOLE camera (no distortion), which is what the 3DGS rasterizer assumes. The undistorted images and the sparse model in PINHOLE convention are saved to `data/gsplat_input/`.

**What COLMAP gives us:**
- For each image: a world-to-camera rotation matrix **R** (3×3) and translation vector **t** (3×1)
- Camera intrinsics: focal lengths fx, fy and principal point cx, cy
- A sparse set of 3D points with known world-space positions

---

## 3. Pipeline A — 3D Gaussian Splatting

### 3.1 What is a 3D Gaussian?

A 3D Gaussian is a volumetric blob defined by:

| Parameter | Symbol | Dimension | What it represents |
|-----------|--------|-----------|-------------------|
| Position (mean) | μ | 3 | World-space center of the blob |
| Rotation | q | 4 | Unit quaternion defining orientation |
| Scale | s | 3 | Half-widths along each local axis (stored as log s) |
| Opacity | α | 1 | How opaque the blob is (stored as logit α) |
| Color (SH) | f | 48 | Spherical harmonic coefficients, degree 3 |

The rotation and scale together define a 3D covariance matrix **Σ = R · S · Sᵀ · Rᵀ**, where R is the rotation matrix and S = diag(s). This covariance describes an ellipsoidal blob — it can be flat like a disc (representing a surface) or round like a ball (representing a fuzzy region).

**Spherical Harmonics for color** encode view-dependent appearance. Degree 3 SH uses 16 coefficients per color channel (48 total). The DC term (degree 0) is view-independent base color; higher-degree terms capture specular highlights and reflections that change with viewing angle. During training, SH degree is progressively increased: degree 0 first (flat color), ramping up to degree 3 as the scene structure stabilizes.

### 3.2 Scene Representation

The entire scene is represented as a **collection of N 3D Gaussians** living in world space. N starts small (seeded from the COLMAP sparse point cloud — in our case ~1,377 points) and grows via densification to typically tens of thousands or hundreds of thousands of Gaussians over the course of training.

There is no mesh, no voxel grid, no neural network implicit function. The Gaussians directly are the scene representation — they are both the storage format and the rendering primitive.

### 3.3 Differentiable Rasterization

To render a view from a given camera pose, the 3DGS rasterizer does the following:

**Step 1 — Project to 2D**
Each 3D Gaussian is projected onto the image plane. The 3D covariance **Σ** is transformed to a 2D covariance **Σ'** in screen space using the Jacobian of the projective transformation. This gives a 2D ellipse on screen for each Gaussian.

**Step 2 — Sort by Depth**
All Gaussians are sorted front-to-back by their projected depth. This is necessary for correct alpha compositing.

**Step 3 — Tile-Based Alpha Compositing**
The image is divided into 16×16 pixel tiles. For each tile, only the Gaussians that overlap it are processed. Pixels are rendered via front-to-back alpha compositing:

```
C = Σᵢ (αᵢ · cᵢ · Πⱼ<ᵢ (1 - αⱼ))
```

where αᵢ is the opacity of Gaussian i weighted by its 2D Gaussian evaluated at the pixel position, cᵢ is the color from the SH evaluation at the viewing direction, and the product term accounts for occlusion by all closer Gaussians.

This is fully **differentiable** — gradients can flow back from pixel color errors through the compositing operation to every Gaussian parameter.

### 3.4 What is Being Trained?

For each of the N Gaussians, we are simultaneously optimizing **all of the following parameters** by gradient descent:

- **μ** (position): where in 3D space does this Gaussian live?
- **q** (rotation): what orientation does the ellipsoid have?
- **s** (scale, log-space): how big is it, and how elongated?
- **α** (opacity, logit-space): how transparent is it?
- **f** (SH coefficients): what color does it appear from each viewing direction?

There is no neural network involved. The model is a direct parametric representation — the parameters listed above are the entire model. Training is pure gradient descent on these ~(N × 59) floating point values.

The optimizer is **Adam** with separate learning rates for each parameter type. Positions use a smaller lr (1.6e-4) since they are in world-space units; SH coefficients use an even smaller lr (2.5e-3 / 20 for higher-degree terms) since they are less critical early in training.

### 3.5 Loss Function

The loss function compares the rendered image R against the ground truth training image GT pixel by pixel:

```
L = (1 - λ) · L1 + λ · L_SSIM
```

With `--ssim-lambda 0.2`, this is **80% L1 + 20% SSIM**.

**L1 loss** (mean absolute error):
```
L1 = (1/HW) Σᵢ |R(i) - GT(i)|
```
This penalizes per-pixel color deviation linearly. It is simple and converges fast but can tolerate blurriness.

**SSIM loss** (Structural Similarity Index):
```
L_SSIM = 1 - SSIM(R, GT)
```
SSIM compares local patches (11×11 window) by measuring three things: luminance similarity, contrast similarity, and structural similarity (correlation of local gradients). It is a perceptual metric — it penalizes blurring and structural distortion more strongly than L1, encouraging the model to reproduce sharp edges and fine detail rather than averaging over uncertainty.

The combination of L1 (pixel accuracy) + SSIM (structural sharpness) is the standard from the original 3DGS paper (Kerbl et al., 2023) and has been shown empirically to outperform either alone.

**Note on LPIPS:** During evaluation steps (not training), an additional **LPIPS** (Learned Perceptual Image Patch Similarity) metric is computed using a pretrained AlexNet. This is a perceptual similarity metric that correlates strongly with human judgement. It is only used for reporting, not in the gradient.

### 3.6 Densification and Pruning

The initial 1,377 Gaussians from the COLMAP sparse cloud are far too few to represent a complex scene. The training procedure adaptively adds and removes Gaussians:

**Densification** (steps 500–5000, every 100 steps):

The key signal is the **2D positional gradient** — how much the rendered position of each Gaussian's projection is being pushed around by the loss. A large accumulated gradient means the Gaussian is in a region that needs more detail.

Two cases:
- **Clone**: if a Gaussian is small (scale below threshold) and has large gradient → copy it and perturb both copies. Used for under-reconstructed fine details.
- **Split**: if a Gaussian is large (scale above threshold) and has large gradient → split into two smaller Gaussians sampled along the principal axis. Used when one big blob is trying to cover two distinct surfaces.

Our thresholds: `grow-grad2d=0.0002`, `grow-scale3d=0.01`.

**Pruning** (every 100 steps after 500):
- Gaussians with opacity below threshold (`prune-scale3d=0.1`) are deleted — they have been driven to transparency because they don't contribute to any training view.
- Gaussians that grow extremely large in screen space are also pruned — they have become degenerate "floaters."

**Opacity Reset** (step 3000):
All opacities are periodically reset to a low value. This forces the optimizer to re-justify each Gaussian's existence — ones that don't help are pruned in the next cycle, preventing a proliferation of useless semi-transparent blobs that degrade novel view quality.

By step 7000, the scene typically contains **100,000–500,000 Gaussians** depending on scene complexity. Each one has learned its position, shape, and view-dependent color to accurately represent a small region of the scene.

### 3.7 Novel View Rendering

After training, rendering a novel view is a **forward pass only** — no optimization, no gradient computation.

1. Provide a new camera pose (rotation + translation) and intrinsics that were not in the training set
2. Run the differentiable rasterizer forward: project all Gaussians, sort by depth, alpha-composite
3. Read off the rendered RGB image

Three trajectory types are used to generate novel views:
- **Ellipse**: a smooth elliptical orbit around the scene centroid at a fixed elevation
- **Interpolated**: a B-spline that passes smoothly through all training camera positions, rendered at 5× temporal density — effectively a smooth slow-motion fly-through
- **Spiral**: a zooming spiral that both orbits and moves forward, good for forward-facing scenes like the robot video

From each trajectory, frames whose optical axis is more than 15° away from every training view are selected as "genuinely novel" — they show geometry that was never directly supervised.

---

## 4. Pipeline B — Diffusion-Based NVS

### 4.1 Zero123++ Overview

Zero123++ (`sudo-ai/zero123plus-v1.2`) is a **2D diffusion model** fine-tuned to perform single-image novel view synthesis. Given one RGB photograph of an object, it generates a 2×3 grid of 6 views simultaneously, each showing the object from a different pre-defined viewpoint (approximately 30°, 90°, 150°, 210°, 270°, 330° azimuth at ~20° elevation).

Unlike the 3DGS pipeline, Zero123++ does **not** build an explicit 3D model. It reasons about geometry implicitly — all 3D understanding is baked into the weights of the neural network during its large-scale training.

### 4.2 How Diffusion Models Work

A **diffusion model** is trained to reverse a noise process. During training:

1. Take a real image x₀
2. Progressively add Gaussian noise over T steps: x₁, x₂, ..., x_T where x_T ≈ N(0, I)
3. Train a neural network (a U-Net with attention layers) to predict the noise ε added at each step, conditioned on a context signal c: ε_θ(xₜ, t, c)

The training loss is:
```
L = E[||ε - ε_θ(xₜ, t, c)||²]
```

This is simply mean squared error between the true noise and the predicted noise — denoising as a regression problem.

At inference time, start from pure noise x_T ~ N(0, I) and iteratively apply the learned denoising:
```
x_{t-1} = (1/√αₜ) · (xₜ - (1-αₜ)/√(1-ᾱₜ) · ε_θ(xₜ, t, c)) + σₜ · z
```
where z ~ N(0, I) adds controlled stochasticity. After T steps of this reverse process, x₀ is a clean generated image consistent with the conditioning c.

**Zero123++ uses the DDIM scheduler** (Denoising Diffusion Implicit Models) with Euler ancestral steps — this allows high-quality generation in 75 steps rather than the 1000 steps needed by the original DDPM formulation.

### 4.3 How Zero123++ Conditions on Viewpoint

The conditioning signal c for Zero123++ has two components:

**1. Input image conditioning:** The reference photograph is encoded by a CLIP image encoder (ViT-L/14). The resulting embedding captures semantic content (what object this is, its shape, materials) and is injected into the U-Net via cross-attention at each resolution level.

**2. Camera conditioning:** Zero123++ generates all 6 views in a single forward pass by concatenating the 6 target view images into a 2×3 grid and running the denoiser on the full grid jointly. The 6 viewpoint angles are fixed and baked into the model — it was trained on 3D object datasets (Objaverse) rendered at exactly these 6 relative camera positions, so the network has learned that "the image in the top-left position should show the object from 30° azimuth."

The key insight is that **cross-view consistency** is enforced because all 6 views are generated simultaneously in the same denoising pass. The network can attend across all 6 view positions, allowing it to make views geometrically consistent with each other (e.g. if a surface is shiny in view 1, it should be shiny from a different angle in view 3).

### 4.4 Inference at Runtime

For each of the 25 selected input frames:

1. **Preprocess**: square-crop the image, composite onto white background, resize to 320×320
2. **Encode**: the CLIP encoder produces a 1024-dimensional embedding of the input photo
3. **Denoise**: start from a 960×640 block of pure Gaussian noise (the 6-view grid size), run 75 steps of the Euler-ancestral reverse diffusion, conditioned on the CLIP embedding
4. **Decode**: the denoised latent is decoded by the VAE decoder to a full-resolution 960×640 RGB image
5. **Split**: the 2×3 grid is cut into 6 individual 320×320 images; we keep the first 4

Total: 25 input images × 4 novel views = **100 generated novel view images**.

---

## 5. Comparison of Approaches

| | 3D Gaussian Splatting | Diffusion NVS (Zero123++) |
|---|---|---|
| **Input required** | 80+ posed images (COLMAP output) | 1 image per novel view query |
| **Training time** | ~30–40 min (GPU) | None (pre-trained model) |
| **Inference time** | Milliseconds per frame | ~2 min per image on T4 |
| **3D consistency** | Fully 3D-consistent (single model) | Best-effort (per-image) |
| **Novel view freedom** | Any viewpoint, any trajectory | 6 fixed relative angles only |
| **Geometric accuracy** | High (ground-truth camera poses) | Approximate (inferred from appearance) |
| **Works on** | Static scenes with good multi-view coverage | Any single object photo |
| **Failure modes** | Floaters in unobserved regions, blur on glass/mirrors | Hallucinated geometry, view inconsistency |
| **Output quality metric** | PSNR / SSIM / LPIPS on held-out views | Perceptual quality (no ground truth) |

**When 3DGS wins:** you have a controlled video with good coverage, you need precise geometry, and you want to fly arbitrary camera paths through the scene.

**When diffusion wins:** you only have a single photo, the scene has objects that are hard to reconstruct (shiny, thin, furry), or you need views of objects not in the training set.

---

## 6. Glossary

**Alpha compositing** — Blending transparent layers front-to-back: each layer contributes its color weighted by its opacity and the remaining transparency of all layers in front of it.

**Bundle adjustment** — Joint nonlinear optimization of all camera poses and 3D point positions to minimize total reprojection error. The backbone of COLMAP's accuracy.

**CLIP** — Contrastive Language-Image Pretraining. A vision encoder trained on 400M image-text pairs. Produces embeddings that capture high-level semantic content of images.

**Covariance matrix** — A symmetric positive definite matrix encoding the shape and orientation of a Gaussian distribution. In 3DGS, the 3D covariance defines how elongated and oriented each ellipsoidal blob is.

**Densification** — The process of adding new Gaussians to under-represented regions during training, guided by gradient magnitude signals.

**Epipolar constraint** — Given two camera poses, a 3D point seen in one image must project to a specific line (the epipolar line) in the other image. COLMAP uses this to filter false feature matches.

**L1 loss** — Mean absolute error between predicted and target values. Robust to outliers compared to L2.

**LPIPS** — Learned Perceptual Image Patch Similarity. Measures perceptual image distance using deep features from a classification network. Correlates better with human perception than PSNR.

**PnP (Perspective-n-Point)** — Algorithm to estimate camera pose given N known 3D world points and their corresponding 2D image projections.

**PSNR** — Peak Signal-to-Noise Ratio. Computed as 10·log₁₀(1/MSE). Higher is better; >25 dB is generally good for NVS, >30 dB is excellent.

**Reprojection error** — The distance in pixels between where a 3D point is observed in an image and where the current camera model predicts it should project. COLMAP minimizes this.

**SfM (Structure from Motion)** — The problem of simultaneously recovering camera poses and scene geometry from a collection of 2D images.

**SIFT (Scale-Invariant Feature Transform)** — A classical algorithm for detecting and describing local image features that are robust to changes in scale, rotation, and illumination.

**Spherical Harmonics** — A set of orthogonal basis functions defined on the sphere. In 3DGS, low-degree SH coefficients encode smooth view-dependent color variation (like how a material looks different from different angles).

**SSIM (Structural Similarity Index)** — A perceptual quality metric that measures luminance, contrast, and structural similarity between image patches. Ranges from -1 to 1 (1 = identical).

**VAE (Variational Autoencoder)** — In latent diffusion models (like the one Zero123++ is built on), the VAE compresses images from pixel space to a smaller latent space where the diffusion process operates, then decodes back to pixels.
