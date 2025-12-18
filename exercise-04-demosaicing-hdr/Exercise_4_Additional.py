import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from PIL.ExifTags import TAGS

# ------------ 1. Settings ------------
folder = "/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/ex4_additional_exercise_data"
image_paths = sorted(glob(os.path.join(folder, "*.JPG")))

# ------------ 2. Estimate exposure times from EXIF ------------
def get_exposure_time(path):
    img = Image.open(path)
    exif = img._getexif()
    for tag, val in exif.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "ExposureTime":
            return float(val)

exposure_times = np.array([get_exposure_time(p) for p in image_paths], dtype=np.float32)
print("Exposure times (s):", exposure_times)

# ------------ 3. Read images in OpenCV format (BGR) ------------
images = [cv2.imread(p) for p in image_paths]

# ------------ 4. Estimate CRF ------------
calibrate = cv2.createCalibrateDebevec()
response_debevec = calibrate.process(images, times=exposure_times)

# ------------ 5. Merge HDR using estimated CRF ------------
merge_debevec = cv2.createMergeDebevec()
hdr_linear = merge_debevec.process(images, times=exposure_times, response=response_debevec)

# ------------ 6. Tone Mapping using Logarithmic Mapping ------------
log_hdr = np.log1p(hdr_linear)
log_min = np.percentile(log_hdr, 0.01)
log_max = np.percentile(log_hdr, 99.99)
log_hdr = (log_hdr - log_min) / (log_max - log_min)
log_hdr = np.clip(log_hdr, 0, 1)

# ------------ 7. Show and Save Result ------------
plt.imshow(cv2.cvtColor((log_hdr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("HDR from JPGs")
plt.axis("off")
plt.show()

output_dir = "/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/results"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "task9_additional_hdr.jpg")
cv2.imwrite(output_path, (log_hdr * 255).astype(np.uint8))