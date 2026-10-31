# Data

## Why not download SEN12MS-CR / M3R-CR directly?
The paper (PSF-CR, IEEE JSTARS 2026) evaluates on **SEN12MS-CR** (mediaTUM 1554803, >400 GB) and **M3R-CR** (gitlab.lrz.de/ai4eo/m3r-cr). Neither host could be reached from the build environment, and SEN12MS-CR is far larger than the available disk. This project therefore **re-creates a SEN12MS-CR-style dataset from the same real, openly licensed Copernicus sources and with the same protocol**. No synthetic or simulated data is used: the clouds, SAR and targets are all real acquisitions.

| Item | SEN12MS-CR (paper) | This project |
|---|---|---|
| SAR | Sentinel-1, VV + VH (dB), 10 m | Sentinel-1 IW GRD, radiometrically terrain-corrected γ⁰, VV + VH (dB), 10 m |
| Cloudy optical | Sentinel-2 L1C, 13 bands, 10 m | Sentinel-2 L1C, 13 bands, resampled to 10 m, acquired ≤ 6 days from the S1 scene |
| Cloud-free target | Sentinel-2 L1C, same season | Sentinel-2 L1C, same tile **and relative orbit**, < 3 % scene cloud, typically 4–10 days from the cloudy image (≤ 45 days) |
| Cloud mask | rule-based detector | same rule-based detector (Schmitt et al. 2019 / SEN12MS-CR toolbox), threshold 0.2 |
| Patch size | 256 × 256 | 256 × 256 |
| Split | geospatially separated test ROIs | split by ROI (whole Sentinel-1 scene), no region shared between train / val / test |

## Sources (all open)
1. **Sentinel-1 RTC** — Google Earth Engine public bucket `gs://earthengine-public/s1/` (ASF HyP3 RTC, GAMMA, 10 m, γ⁰ power; 10 globally distributed 2019 scenes). Contains modified Copernicus Sentinel data 2019.
2. **Sentinel-2 L1C** — Google Cloud public archive `gs://gcp-public-data-sentinel-2/tiles/...` (index `index.csv.gz`). Copernicus Sentinel data 2018–2019.

## Final dataset (`raw/sen12mscr_style/`)
* **1,149 triplets** from 8 regions of interest (Cambodia ×2 scenes, Costa Rica/Panama, Bangladesh, Nepal, Ecuador, Amazonas/Brazil, Rondônia/Brazil–Bolivia border), 21 cloudy/cloud-free Sentinel-2 pairs over 19 MGRS tiles.
* `s1/<patch>.tif` — 2 bands float32 (VV, VH in dB).
* `s2_cloudy/<patch>.tif` — 13 bands uint16 (B01…B12 incl. B8A, L1C DN).
* `s2/<patch>.tif` — 13 bands uint16, cloud-free target.
* `metadata.csv` — patch id, ROI, MGRS tile, S1 scene, S2 product ids, dates, cloud fraction, lon/lat, CRS.
* All GeoTIFFs keep their UTM georeferencing.

Quality filters applied when extracting: no Sentinel-2 nodata, < 0.1 % SAR nodata, cloudy image 5–98 % cloud (rule-based mask), target ≤ 2 % cloud.

## Processed data (`processed/`)
Created by the training notebook: normalised float16 arrays per split (`{train,val,test}_{sar,opt,mask,tgt}.npy`) and `{split}_index.csv`.
Normalisation (paper Sec. IV-A-2): VV clipped to [-25, 0] dB, VH to [-32.5, 0] dB, Sentinel-2 to [0, 10000]; all scaled to [0, 1].
Split (by ROI): 781 train / 114 validation / 254 test patches.

## Rebuilding the dataset
Requires the Google Cloud SDK (`gsutil`, anonymous access works) and `rasterio`:
```bash
cd data/scripts
gsutil cat gs://gcp-public-data-sentinel-2/index.csv.gz | zcat | python filt_index.py > s2_candidates.csv   # Sentinel-2 granules overlapping the S1 scenes
python pairs.py                                                                                      # cloudy / cloud-free pairing -> s2_pairs.csv
python build_dataset.py ../raw/sen12mscr_style /tmp/s2_download 110                                  # extract 256x256 triplets
```
