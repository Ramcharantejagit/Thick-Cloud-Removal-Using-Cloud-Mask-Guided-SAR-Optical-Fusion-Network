"""Cloud-removal inference with the saved best model (models/best_model).

Mirrors code/02_Best_Model_Prediction.ipynb: same preprocessing (from preprocessing.json),
same TorchScript model, same test-time-augmentation uncertainty.
"""
import json
import os
import threading
import time
import warnings

import numpy as np
import rasterio
import torch
from PIL import Image
from scipy import ndimage, signal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_DIR = os.path.join(PROJECT_ROOT, "models", "best_model")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "sen12mscr_style")
TEST_INDEX = os.path.join(PROJECT_ROOT, "data", "processed", "test_index.csv")
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "model_evaluation_results")

with open(os.path.join(BEST_DIR, "preprocessing.json")) as f:
    PREPROC = json.load(f)
with open(os.path.join(BEST_DIR, "best_model_config.json")) as f:
    MODEL_CFG = json.load(f)

_model, _lock = None, threading.Lock()


class InputError(ValueError):
    """Raised for invalid user input rasters."""


def get_model():
    global _model
    with _lock:
        if _model is None:
            torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                _model = torch.jit.load(os.path.join(BEST_DIR, "best_model_torchscript.pt"), map_location="cpu").eval()
    return _model


# ---------------- preprocessing (identical to training) ----------------
def cloud_score(dn, cfg=PREPROC):
    d = dn.astype("float32") / 10000.0
    r = lambda x, lo_hi: (x - lo_hi[0]) / (lo_hi[1] - lo_hi[0])
    R = cfg["cloud_rules"]
    s = np.ones(d.shape[1:], "float32")
    s = np.minimum(s, r(d[1], R["B02"]))
    s = np.minimum(s, r(d[0], R["B01"]))
    s = np.minimum(s, r(d[0] + d[10], R["B01+B10"]))
    s = np.minimum(s, r(d[3] + d[2] + d[1], R["B02+B03+B04"]))
    ndsi = (d[2] - d[11]) / (d[2] + d[11] + 1e-6)
    s = np.minimum(s, r(ndsi, R["NDSI(B03,B11)"]))
    k, b = cfg["cloud_closing_size"], cfg["cloud_box_size"]
    s = ndimage.grey_closing(s, size=(k, k))
    s = signal.convolve2d(s, np.ones((b, b)) / b ** 2, mode="same")
    return np.clip(s, 1e-5, 1.0)


def preprocess(s1_db, s2_dn, cfg=PREPROC):
    lo = np.array([cfg["s1_clip_db"][b][0] for b in cfg["s1_bands"]], "float32")[:, None, None]
    hi = np.array([cfg["s1_clip_db"][b][1] for b in cfg["s1_bands"]], "float32")[:, None, None]
    sar = (np.clip(s1_db, lo, hi) - lo) / (hi - lo)
    a, b = cfg["s2_clip"]
    opt = (np.clip(s2_dn.astype("float32"), a, b) - a) / (b - a)
    mask = (cloud_score(s2_dn, cfg) >= cfg["cloud_threshold"]).astype("float32")[None]
    return sar.astype("float32"), opt.astype("float32"), mask


def normalise_reference(ref_dn, cfg=PREPROC):
    a, b = cfg["s2_clip"]
    return (np.clip(ref_dn.astype("float32"), a, b) - a) / (b - a)


# ---------------- input reading + validation ----------------
def read_inputs(s1_path, s2_path, ref_path=None):
    size = tuple(MODEL_CFG["input_size"])
    try:
        with rasterio.open(s1_path) as a, rasterio.open(s2_path) as b:
            if a.count != 2:
                raise InputError(f"Sentinel-1 file must have 2 bands (VV, VH in dB); got {a.count}.")
            if b.count != 13:
                raise InputError(f"Sentinel-2 file must have 13 bands (L1C B01–B12 incl. B8A); got {b.count}.")
            if a.shape != size or b.shape != size:
                raise InputError(f"Both rasters must be {size[0]}×{size[1]} pixels; got S1 {a.shape}, S2 {b.shape}.")
            if a.crs != b.crs or not a.transform.almost_equals(b.transform):
                raise InputError("Sentinel-1 and Sentinel-2 rasters must be co-registered (same CRS and transform).")
            s1, s2, profile = a.read().astype("float32"), b.read(), b.profile
        ref = None
        if ref_path:
            with rasterio.open(ref_path) as c:
                if c.count != 13 or c.shape != size:
                    raise InputError("Reference (cloud-free) file must be a 13-band raster of the same size.")
                ref = c.read()
    except rasterio.errors.RasterioIOError:
        raise InputError("File is not a readable GeoTIFF.")
    if not np.isfinite(s1).all():
        raise InputError("Sentinel-1 raster contains NaN/Inf values.")
    if s1.min() < -80 or s1.max() > 40:
        raise InputError("Sentinel-1 values do not look like backscatter in dB (expected roughly -50…+10).")
    if s2.max() <= 1.5:
        raise InputError("Sentinel-2 values look like reflectance in [0,1]; L1C digital numbers (0–10000) are expected.")
    return s1, s2, profile, ref


# ---------------- prediction ----------------
@torch.no_grad()
def predict(sar, opt, mask, tta=True):
    model = get_model()
    x = [torch.from_numpy(v)[None] for v in (sar, opt, mask)]
    outs = []
    for k in (range(4) if tta else [0]):
        for flip in ((False, True) if tta else (False,)):
            t = [torch.rot90(v, k, (2, 3)) for v in x]
            t = [v.flip(3) for v in t] if flip else t
            y = model(*t)
            y = y.flip(3) if flip else y
            outs.append(torch.rot90(y, -k, (2, 3)))
    outs = torch.stack(outs)
    unc = outs.std(0)[0].mean(0).numpy() if len(outs) > 1 else np.zeros(opt.shape[1:], "float32")
    return outs.mean(0)[0].numpy(), unc


def metrics(p, t):
    mse = float(((p - t) ** 2).mean())
    cos = (p * t).sum(0) / (np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + 1e-8)
    return {"psnr": float(10 * np.log10(1 / max(mse, 1e-10))),
            "sam": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()),
            "mae": float(np.abs(p - t).mean())}


# ---------------- visualisation / saving ----------------
def _rgb(x):
    return (np.clip(x[[3, 2, 1]].transpose(1, 2, 0) * 3.3, 0, 1) * 255).astype("uint8")


def _gray(x, lo=None, hi=None):
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    return (np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1) * 255).astype("uint8")


def _heat(x, vmax):
    v = np.clip(x / max(vmax, 1e-8), 0, 1)
    r = np.clip(1.5 * v, 0, 1); g = np.clip(1.5 * v - 0.5, 0, 1); b = np.clip(0.6 - v, 0, 1) * 0.8
    return (np.stack([r, g, b], -1) * 255).astype("uint8")


def run(s1_path, s2_path, out_dir, ref_path=None, tta=True):
    """Full pipeline. Returns a dict of results; writes PNGs + GeoTIFFs to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    s1, s2, profile, ref_dn = read_inputs(s1_path, s2_path, ref_path)
    sar, opt, mask = preprocess(s1, s2)
    t0 = time.time()
    pred, unc = predict(sar, opt, mask, tta=tta)
    runtime_ms = (time.time() - t0) * 1000
    conf_map = 1.0 - np.clip(unc / 0.05, 0, 1)
    m = mask[0] > 0.5
    res = {"cloud_cover": float(mask.mean()), "runtime_ms": runtime_ms, "tta": bool(tta),
           "confidence": float(conf_map.mean()) if tta else None,
           "confidence_cloud": float(conf_map[m].mean()) if (tta and m.any()) else None,
           "mean_uncertainty": float(unc.mean()) if tta else None,
           "psnr_input": None, "psnr_pred": None, "sam_input": None, "sam_pred": None, "mae_input": None, "mae_pred": None}
    images = {"cloudy": _rgb(opt), "mask": _gray(mask[0], 0, 1), "sar": _gray(sar[0], 0, 1), "prediction": _rgb(pred)}
    if tta:
        images["uncertainty"] = _heat(unc, max(float(np.percentile(unc, 99)), 1e-3))   # stretched to the 99th percentile
    if ref_dn is not None:
        ref = normalise_reference(ref_dn)
        mi, mp = metrics(opt, ref), metrics(pred, ref)
        res.update({"psnr_input": mi["psnr"], "psnr_pred": mp["psnr"], "sam_input": mi["sam"], "sam_pred": mp["sam"],
                    "mae_input": mi["mae"], "mae_pred": mp["mae"]})
        images["reference"] = _rgb(ref)
    for k, im in images.items():
        Image.fromarray(im).save(os.path.join(out_dir, f"{k}.png"))
    prof = {**profile, "count": 13, "dtype": "uint16", "compress": "deflate"}
    prof.pop("nodata", None)
    with rasterio.open(os.path.join(out_dir, "cloudfree.tif"), "w", **prof) as dst:
        dst.write(np.round(pred * PREPROC["s2_clip"][1]).astype("uint16"))
    res["images"] = sorted(images)
    return res


# ---------------- project data ----------------
def sample_patches():
    """Held-out test patches (never used for training) available as samples."""
    import pandas as pd
    if not os.path.exists(TEST_INDEX):
        return []
    df = pd.read_csv(TEST_INDEX)
    ok = df.patch.map(lambda p: all(os.path.exists(os.path.join(RAW_DIR, d, p + ".tif")) for d in ("s1", "s2_cloudy", "s2")))
    df = df[ok].sort_values("patch")
    return df[["patch", "roi", "tile", "cloudy_date", "clear_date", "cloud_frac", "lon", "lat"]].to_dict("records")


def sample_paths(patch):
    return tuple(os.path.join(RAW_DIR, d, patch + ".tif") for d in ("s1", "s2_cloudy", "s2"))


def benchmark():
    """Actual test results of all trained models (from the training notebook)."""
    import pandas as pd
    p = os.path.join(EVAL_DIR, "test_results.csv")
    r = os.path.join(EVAL_DIR, "region_results.csv")
    out = {"results": pd.read_csv(p).to_dict("records") if os.path.exists(p) else [],
           "regions": pd.read_csv(r).to_dict("records") if os.path.exists(r) else []}
    return out
