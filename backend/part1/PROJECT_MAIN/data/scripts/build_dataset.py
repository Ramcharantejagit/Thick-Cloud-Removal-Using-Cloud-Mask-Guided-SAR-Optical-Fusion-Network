"""Build a SEN12MS-CR-style triplet dataset (S1 SAR, S2 cloudy, S2 cloud-free) from public Copernicus data.
S1: ASF HyP3 RTC gamma0 (10 m, VV/VH) in gs://earthengine-public/s1
S2: Level-1C 13-band products in gs://gcp-public-data-sentinel-2
"""
import os, sys, subprocess, glob, shutil, json, numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy import ndimage, signal
os.environ.setdefault('CURL_CA_BUNDLE', '/root/.ccr/ca-bundle.crt')
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_HTTP_MAX_RETRY'] = '5'
BANDS = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
OUT = sys.argv[1]; TMP = sys.argv[2]; PER_PAIR = int(sys.argv[3]); SEED = 0
PS = 256

def cloud_score(d):
    """Rule-based cloud detector (Schmitt et al. 2019, as used in the SEN12MS-CR/DSen2-CR toolbox). d: (13,H,W) DN."""
    d = d.astype('float32') / 10000.
    r = lambda x, lo, hi: (x - lo) / (hi - lo)
    s = np.ones(d.shape[1:], 'float32')
    s = np.minimum(s, r(d[1], .1, .5)); s = np.minimum(s, r(d[0], .1, .3))
    s = np.minimum(s, r(d[0] + d[10], .15, .2)); s = np.minimum(s, r(d[3] + d[2] + d[1], .2, .8))
    ndsi = (d[2] - d[11]) / (d[2] + d[11] + 1e-6); s = np.minimum(s, r(ndsi, .8, .6))
    s = ndimage.grey_closing(s, size=(5, 5)); s = signal.convolve2d(s, np.ones((7, 7)) / 49., mode='same')
    return np.clip(s, 1e-5, 1.)

def s1_url(stamp_dir):
    stamp = stamp_dir.split('_')[4]
    base = f'/vsicurl/https://storage.googleapis.com/earthengine-public/s1/{stamp_dir}/S1A_IW_RT10_{stamp}_G_gpn_'
    return base + 'VV.tif', base + 'VH.tif'

def download(base_url, dst):
    os.makedirs(dst, exist_ok=True)
    subprocess.run(['gsutil', '-q', '-m', 'cp', f'{base_url}/GRANULE/*/IMG_DATA/*_B*.jp2', dst], check=True)
    files = {b: glob.glob(f'{dst}/*_{b}.jp2')[0] for b in BANDS}
    return files

def read_lowres(files, n=1830):
    arr = []
    for b in BANDS:
        with rasterio.open(files[b]) as src: arr.append(src.read(1, out_shape=(n, n), resampling=Resampling.nearest))
    return np.stack(arr)

def read_patch(files, r0, c0):
    arr = []
    for b in BANDS:
        with rasterio.open(files[b]) as src:
            f = 10980 // src.width  # 1, 2 or 6
            w = Window(c0 // f, r0 // f, PS // f, PS // f) if f > 1 else Window(c0, r0, PS, PS)
            x = src.read(1, window=w)
            if f > 1: x = np.kron(x, np.ones((f, f), x.dtype))[:PS, :PS] if (PS % f == 0) else ndimage.zoom(x, PS / x.shape[0], order=0)[:PS, :PS]
            arr.append(x)
    return np.stack(arr)

def main():
    pairs = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's2_pairs.csv'))
    s1dirs = {l.strip().rstrip('/').split('/')[-1].split('_')[4]: l.strip().rstrip('/').split('/')[-1] for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's1_scenes.txt'))}
    pairs['S1'] = pairs.S1.astype(str)
    pairs = pairs.sort_values(['S1', 'gap_days', 'size'], ascending=[True, True, False])
    sel = pairs.groupby('S1').head(5)  # up to five tile pairs per S1 scene (ROI)
    rng = np.random.default_rng(SEED); meta = []
    for d in ['s1', 's2_cloudy', 's2']: os.makedirs(f'{OUT}/{d}', exist_ok=True)
    done = set(pd.read_csv(f'{OUT}/metadata.csv').pair_id) if os.path.exists(f'{OUT}/metadata.csv') else set()
    if done: meta = pd.read_csv(f'{OUT}/metadata.csv').to_dict('records')
    for _, p in sel.iterrows():
        pid = f"{p.S1}_{p.tile}_{p.cloudy.split('_')[2][:8]}"
        if pid in done: continue
        print('PAIR', pid, flush=True)
        stamp_dir = s1dirs[p.S1]; vv_u, vh_u = s1_url(stamp_dir)
        fc = download(p.cloudy, f'{TMP}/cloudy'); ff = download(p.clear, f'{TMP}/clear')
        with rasterio.open(fc['B02']) as ref: crs, tr = ref.crs, ref.transform
        lc, lf = read_lowres(fc), read_lowres(ff)
        cs_c, cs_f = cloud_score(lc) >= 0.2, cloud_score(lf) >= 0.2
        nod = (lc == 0).any(0) | (lf == 0).any(0)
        with rasterio.open(vv_u) as s1src:
            s1b = transform_bounds(s1src.crs, crs, *s1src.bounds)
            # low-res S1 validity on the S2 60 m grid
            s1low = np.zeros((1830, 1830), 'float32')
            reproject(rasterio.band(s1src, 1), s1low, dst_transform=tr * tr.scale(6, 6), dst_crs=crs, resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        cands = []
        k = PS * 10 // 60  # ~42 lowres px per patch
        for i in range(10980 // PS):
            for j in range(10980 // PS):
                r0, c0 = i * PS, j * PS; a, b = r0 // 6, c0 // 6
                if nod[a:a + k + 1, b:b + k + 1].any() or (s1low[a:a + k + 1, b:b + k + 1] <= 0).any(): continue
                cc, cf = cs_c[a:a + k + 1, b:b + k + 1].mean(), cs_f[a:a + k + 1, b:b + k + 1].mean()
                if 0.05 <= cc <= 0.95 and cf <= 0.02: cands.append((r0, c0, cc))
        print('  candidates', len(cands), flush=True)
        if not cands: shutil.rmtree(TMP, ignore_errors=True); continue
        idx = rng.permutation(len(cands))[:PER_PAIR]
        n_ok = 0
        with rasterio.open(vv_u) as vv, rasterio.open(vh_u) as vh:
            for t in idx:
                r0, c0, _ = cands[t]
                xc, xf = read_patch(fc, r0, c0), read_patch(ff, r0, c0)
                if (xc == 0).any() or (xf == 0).any(): continue
                ptr = tr * rasterio.Affine.translation(c0, r0)
                s1 = np.zeros((2, PS, PS), 'float32')
                for q, src in enumerate((vv, vh)):
                    reproject(rasterio.band(src, 1), s1[q], dst_transform=ptr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
                if (s1 <= 0).mean() > 0.001: continue
                s1 = 10 * np.log10(np.maximum(s1, 1e-6))  # gamma0 power -> dB
                mc, mf = cloud_score(xc) >= 0.2, cloud_score(xf) >= 0.2
                if mf.mean() > 0.02 or not (0.05 <= mc.mean() <= 0.98): continue
                name = f'{pid}_p{r0 // PS:02d}{c0 // PS:02d}'
                prof = dict(driver='GTiff', height=PS, width=PS, crs=crs, transform=ptr, compress='deflate', tiled=False)
                with rasterio.open(f'{OUT}/s1/{name}.tif', 'w', count=2, dtype='float32', predictor=3, **prof) as o: o.write(s1)
                with rasterio.open(f'{OUT}/s2_cloudy/{name}.tif', 'w', count=13, dtype='uint16', predictor=2, **prof) as o: o.write(xc)
                with rasterio.open(f'{OUT}/s2/{name}.tif', 'w', count=13, dtype='uint16', predictor=2, **prof) as o: o.write(xf)
                cx, cy = ptr * (PS / 2, PS / 2); lon, lat = transform_bounds(crs, 'EPSG:4326', cx, cy, cx, cy)[:2]
                meta.append(dict(patch=name, pair_id=pid, roi=p.S1, tile=p.tile, s1_scene=stamp_dir, s2_cloudy=p.cloudy.split('/')[-1], s2_clear=p.clear.split('/')[-1],
                                 cloudy_date=p.cloudy.split('_')[2][:8], clear_date=p.clear.split('_')[2][:8], cloud_frac=float(mc.mean()), lon=lon, lat=lat, crs=str(crs)))
                n_ok += 1
        print('  saved', n_ok, flush=True)
        pd.DataFrame(meta).to_csv(f'{OUT}/metadata.csv', index=False)
        shutil.rmtree(TMP, ignore_errors=True)
    print('DONE', len(meta))
main()
