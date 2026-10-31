# Thick Cloud Removal with Cloud-Mask-Guided SAR-Optical Fusion (PSF-CR and advanced variants)

## Problem statement
Clouds hide large parts of optical satellite images. Sentinel-1 SAR can see through clouds but looks very different from optical imagery and contains speckle noise. The task is to **reconstruct a cloud-free 13-band Sentinel-2 image** from a cloudy Sentinel-2 image, a co-registered Sentinel-1 SAR image (VV, VH) and a cloud mask. This is pixel-wise image regression.

## IEEE paper
S. Meng, S. Li, X. Wang, J. Yang, G. Song, Y. Ding, W. Gong, "**Thick Cloud Removal Using Cloud Mask-Guided SAR-Optical Fusion Network**", *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 19, pp. 4373–4388, 2026. DOI: 10.1109/JSTARS.2026.3651496 (IEEE Xplore 11339871).

Paper method (PSF-CR):
* **SFFA**, joint spatial-frequency feature alignment:
  * **FGCA**: FFT-based global cross-attention between SAR and optical features.
  * **SLCA**: 8×8 window cross-attention in which the SAR and optical streams exchange queries.
* **CMGR**: cloud-mask-guided rectification, a soft sigmoid gating weight matrix W applied to the fused features.
* **Reconstruction U-Net** with 5 stages. Each block is MCL (multi-scale conv) → SLSA (window self-attention) → FGSA (frequency-domain global self-attention).
* **Composite loss**: λ1·(W-weighted L1 + 0.1·(1−SSIM)) + λ2·(L1 on the FFT real and imaginary parts + spectral convergence), with λ1 = 1 and λ2 = 0.1.
* **Training setup**: Adam (lr 2e-4, betas 0.9/0.999, weight decay 1e-4) with a cosine schedule down to 2e-6. Inputs are clipped as follows: VV to [-25, 0] dB, VH to [-32.5, 0] dB, Sentinel-2 to [0, 10000].
* **Metrics**: PSNR, SSIM, SAM and MAE.

## Dataset
The paper uses SEN12MS-CR (mediaTUM 1554803) and M3R-CR (gitlab.lrz.de/ai4eo/m3r-cr). Neither host could be reached from the build environment, and SEN12MS-CR is more than 400 GB.

This project therefore **rebuilds a SEN12MS-CR-style dataset from the same real, open Copernicus sources and with the same protocol**:
* Sentinel-1 RTC VV/VH from `gs://earthengine-public/s1`.
* Sentinel-2 L1C cloudy and cloud-free images from the Google Cloud Sentinel-2 archive.
* Cloud masks from the same rule-based detector used by the SEN12MS-CR toolbox.

The result is **1,149 real 256×256 triplets from 8 regions** (Cambodia, Costa Rica/Panama, Bangladesh, Nepal, Ecuador, Brazil, Brazil–Bolivia border). They are split by region into **781 train, 114 validation and 254 test** patches, so no region appears in more than one split. Details and rebuild scripts are in `data/README.md` and `data/scripts/`.

## Proposed system
The paper's baselines and its own method are implemented, plus more advanced models built on them:

| # | Model | Type | Description |
|---|---|---|---|
| 1 | **DSen2-CR** | existing baseline (in the paper's comparison) | Residual ResNet. SAR and optical are fused at the input; a long skip connection carries the cloudy image (64 features, 8 blocks for CPU training) |
| 2 | **PSF-CR** | paper method | Full SFFA (FGCA + SLCA) + CMGR + MCL/SLSA/FGSA U-Net with the composite loss, as described in the paper |
| 3 | **Restormer-CR** | advanced transformer | Restormer (CVPR 2022): MDTA transposed attention and gated-Dconv FFN. SAR, optical and mask are fused at the input |
| 4 | **PSF-CR++** | proposed advanced model | PSF-CR plus four changes listed below |

PSF-CR++ adds four things to PSF-CR:
1. **LayerNorm-normalised frequency attention** (FSAS-style) in place of the N-scaled spatial softmax. It trained faster in a pilot run on this data.
2. **Gated-Dconv FFNs** in the reconstruction blocks.
3. **Two MDTA transposed-attention blocks** in the bottleneck.
4. A **mask-aware adaptive output fusion**: α·restored + (1−α)·cloudy.

Every model predicts a residual on top of the cloudy image, and its output head starts at zero, so training begins from the cloudy input. All models share the same data, splits, loss, optimiser and training budget.

## Folder structure
```text
PROJECT_MAIN/
├── code/
│   ├── 01_Data_Preprocessing_and_Model_Training.ipynb   # EDA → cleaning → preprocessing → 4 models → evaluation → best model
│   └── 02_Best_Model_Prediction.ipynb                   # standalone inference with the saved best model only
├── data/
│   ├── raw/sen12mscr_style/{s1,s2_cloudy,s2}/*.tif + metadata.csv
│   ├── processed/       # normalised float16 arrays + split index files, predictions/
│   ├── scripts/         # dataset build scripts (public archives → triplets)
│   └── README.md
├── models/
│   ├── dsen2cr/ psfcr/ restormer_cr/ psfcr_plus/   # <key>.pth, model_config.json, preprocessing.json, training_history.csv, test_metrics.json
│   ├── best_model/      # best_model_torchscript.pt, best_model_state_dict.pth, best_model_config.json, preprocessing.json
│   └── model_evaluation_results/   # test_results.csv, region_results.csv, per_band_psnr.csv, psnr_vs_coverage.csv, plots, per-image metrics
├── webapp/              # Flask web app (app.py, inference.py, templates/, static/, tests/)
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```
Tested with Python 3.13, torch 2.14 (CPU), rasterio 1.5 and pytorch-msssim 1.0.

## How to run the training notebook
```bash
cd code
jupyter notebook 01_Data_Preprocessing_and_Model_Training.ipynb   # or: jupyter nbconvert --to notebook --execute --inplace 01_Data_Preprocessing_and_Model_Training.ipynb
```
* The notebook reads `data/raw/`, writes `data/processed/`, trains the 4 models, writes `models/` and selects and exports the best model.
* A full run takes about 4.5 h on 2 CPU cores and is much faster on a GPU, which is used automatically if present.
* Training settings are in the configuration cell (`TRAIN_CFG`).

## How to run the prediction notebook
```bash
cd code
jupyter notebook 02_Best_Model_Prediction.ipynb
```
* The notebook loads only `models/best_model/` and applies the preprocessing stored in `preprocessing.json`.
* It runs on a sample test patch from a held-out region. To use your own data, point `S1_PATH` and `S2_CLOUDY_PATH` to your own co-registered 256×256 GeoTIFFs: S1 with VV and VH in dB, S2 L1C with 13 bands.
* It produces:
  * the cloud-free 13-band GeoTIFF;
  * an uncertainty map from 8-view test-time augmentation and an overall confidence score;
  * PSNR/SAM/MAE against the reference image, when one is available.

## Web application (CloudClear)
`webapp/` contains a Flask web app with user registration, admin approval, a dashboard, cloud removal with the saved best model, and history. Run it with:
```bash
cd webapp && pip install -r requirements.txt && python app.py   # http://127.0.0.1:5000
```
The default admin login is `admin` / `Admin@123`. See `webapp/README.md` for details.

## Model results (actual test results, 254 patches, 2 held-out regions)
| Model | PSNR ↑ | SSIM ↑ | SAM (°) ↓ | MAE ↓ | Params (M) | CPU ms/img |
|---|---|---|---|---|---|---|
| Cloudy input (no model) | 20.91 | 0.737 | 12.71 | 0.0784 | – | – |
| DSen2-CR | 29.12 | 0.862 | **8.13** | 0.0282 | 0.61 | 407 |
| **PSF-CR (paper)** | **30.08** | **0.864** | 8.44 | 0.0245 | 3.30 | 852 |
| Restormer-CR | 26.77 | 0.828 | 10.65 | 0.0379 | 1.16 | 442 |
| PSF-CR++ (proposed) | 29.98 | 0.858 | 8.30 | **0.0244** | 3.67 | 845 |

Cloudy vs cloud-free regions (PSNR, dB), as in the paper's Table III:

| Model | Cloud region | Clear region |
|---|---|---|
| Cloudy input | 18.31 | 24.36 |
| DSen2-CR | 28.33 | 29.72 |
| PSF-CR | 29.33 | **30.74** |
| Restormer-CR | 25.97 | 27.48 |
| PSF-CR++ | **29.37** | 30.54 |

More breakdowns are in `models/model_evaluation_results/`: per-band PSNR, PSNR against cloud coverage, training curves and qualitative comparisons.

## Best model
**PSF-CR (paper method)** has the best average rank over PSNR, SSIM, SAM and MAE. It is exported to `models/best_model/` both as TorchScript (self-contained, so the prediction notebook needs no model class code) and as a state dict.

On the sample prediction patch (50.7 % cloud), PSNR rises from 21.11 dB for the cloudy input to 29.72 dB for the prediction.

## Conclusion
* SAR-optical fusion removes thick clouds effectively on real data from unseen regions: +9.2 dB PSNR and a 69 % lower MAE than the cloudy input.
* The paper's PSF-CR gives the best overall quality (+0.96 dB PSNR over DSen2-CR), which confirms that explicit SAR-optical alignment and cloud-mask guidance help.
* The proposed PSF-CR++ is almost tied with it. It has the lowest MAE and the best PSNR inside clouds, but slightly lower overall PSNR and SSIM, so with this training budget it did not beat the paper's model.
* Restormer-CR, a strong general restoration model without explicit cross-modal alignment, performs worst. This is consistent with the paper's argument.

**Limitations:**
* Training was CPU-only: 6 epochs, 128×128 crops, batch 4, against the paper's 50 epochs on an RTX 4090. Absolute values are therefore not directly comparable to the paper's Table I.
* The dataset is a SEN12MS-CR-style rebuild with about 1.1k patches, not the original 122k.
