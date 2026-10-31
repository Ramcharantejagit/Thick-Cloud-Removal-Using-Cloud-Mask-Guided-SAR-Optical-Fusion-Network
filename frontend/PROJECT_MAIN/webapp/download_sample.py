
import os
import numpy as np
import rasterio
from datasets import load_dataset

OUTPUT_DIR = "sample_inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Connecting to SEN12MS-CR dataset...")

ds = load_dataset(
    "Hermanni/sen12mscr",
    split="test",
    streaming=True
)

print("Downloading one matching sample...")
sample = next(iter(ds))

# Read Sentinel-1
sar_shape = tuple(sample["sar_shape"])
sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sar_shape)

print("Original SAR shape:", sar.shape)

if sar.ndim != 3:
    raise ValueError(f"Unexpected SAR shape: {sar.shape}")

if sar.shape[-1] == 2:
    sar = np.transpose(sar, (2, 0, 1))
elif sar.shape[0] == 2:
    pass
else:
    raise ValueError(f"Expected 2 SAR bands, got shape {sar.shape}")

print("Corrected SAR shape:", sar.shape)

# Read Sentinel-2 images
opt_shape = tuple(sample["opt_shape"])

cloudy = np.frombuffer(
    sample["cloudy"], dtype=np.int16
).reshape(opt_shape)

target = np.frombuffer(
    sample["target"], dtype=np.int16
).reshape(opt_shape)

if cloudy.shape[-1] == 13:
    cloudy = np.transpose(cloudy, (2, 0, 1))
    target = np.transpose(target, (2, 0, 1))

def save_tif(filename, data):
    path = os.path.join(OUTPUT_DIR, filename)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        compress="deflate"
    ) as dst:
        dst.write(data)

    print(f"Saved: {path}")
    print(f"Bands: {data.shape[0]}")
    print(f"Dimensions: {data.shape[2]} x {data.shape[1]}")

save_tif("sentinel1.tif", sar)
save_tif("cloudy_sentinel2.tif", cloudy)
save_tif("cloudfree_sentinel2.tif", target)

print("\nAll three TIFF files created successfully!")
print("Sample ID:", sample.get("patch", "Unknown"))
