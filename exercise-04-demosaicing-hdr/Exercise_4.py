import numpy as np
import matplotlib.pyplot as plt
import os
import rawpy
from scipy.ndimage import convolve
from glob import glob
import cv2

# Create results folder
output_dir = "/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/results"
os.makedirs(output_dir, exist_ok=True)


# ----------- Task 1:   Investigating Bayer Patterns -----------

# Path to Task 1 image
data_dir = "/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/exercise_4_data/01"
img_path = os.path.join(data_dir, "IMG_9939.npy")

# Load the raw Bayer image
bayer = np.load(img_path)

# Show the whole image
plt.figure(figsize=(6, 6))
plt.imshow(bayer, cmap='gray')
plt.savefig(os.path.join(output_dir, "task1_raw_bayer.png"), bbox_inches='tight')
plt.title("Raw Bayer Image")
plt.axis("off")
plt.show()

# Print a small patch to manually inspect the Bayer layout
patch = bayer[:6, :6]
print("Top-left 6x6 patch:")
print(patch)

"""
Lets focus on first 4×4 block,
We assumed:
- High values are typically Red or Green
- Low values are typically Blue

Then we observed:
Row 0: High, Low, High, Low → Likely R G R G
Row 1: Low, High, Low, High → Likely G B G B

This gives,
R G
G B
Hence, the Bayern pattern is RGGB.

"""


# ----------- Task 2:  Implementing a Demosaicing Algorithm -----------

# ----------- 1. Load raw image from .CR3 -----------
raw = rawpy.imread("/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/exercise_4_data/02/IMG_4782.CR3")
bayer = np.array(raw.raw_image_visible, dtype=np.float32)

# ----------- 2. Separate channels using RGGB pattern -----------
def extract_channels_RGGB(bayer):
    R = np.zeros_like(bayer)
    G = np.zeros_like(bayer)
    B = np.zeros_like(bayer)
    # RGGB pattern
    R[0::2, 0::2] = bayer[0::2, 0::2]
    G[0::2, 1::2] = bayer[0::2, 1::2]
    G[1::2, 0::2] = bayer[1::2, 0::2]
    B[1::2, 1::2] = bayer[1::2, 1::2]
    return R, G, B

# ----------- 3. Interpolate missing pixels with a 3x3 kernel -----------
def interpolate(channel):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32)
    kernel = kernel / kernel.sum()
    mask = (channel > 0).astype(np.float32)
    filtered = convolve(channel, kernel)
    norm = convolve(mask, kernel)
    norm[norm == 0] = 1  # Avoid division by zero
    return filtered / norm

# ----------- 4. Apply interpolation to each channel -----------
R, G, B = extract_channels_RGGB(bayer)
R_interp = interpolate(R)
G_interp = interpolate(G)
B_interp = interpolate(B)

# ----------- 5. Stack into RGB image -----------
rgb_image = np.stack((R_interp, G_interp, B_interp), axis=-1)

# ----------- 6. Visualize (with normalization for display only) -----------
plt.figure(figsize=(8, 8))
plt.imshow(np.clip(rgb_image / rgb_image.max(), 0, 1))
plt.savefig(os.path.join(output_dir, "task2_demosaiced_rgb.png"), bbox_inches='tight')
plt.title("Demosaiced RGB (Linear)")
plt.axis("off")
plt.show()


# ----------- Task 3:  Gamma Correction -----------

def gamma_correction(data, gamma):
    a = np.percentile(data, 0.01)
    b = np.percentile(data, 99.99)
    norm = (data - a) / (b - a)
    norm = np.clip(norm, 0, 1)
    return norm ** gamma

gamma_img = gamma_correction(rgb_image, gamma=0.3)
alt_gamma_img = gamma_correction(rgb_image, gamma=0.5)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gamma_img)
plt.title("Gamma 0.3")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(alt_gamma_img)
plt.title("Gamma 0.5 (Alt)")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "task3_gamma_comparison.png"), bbox_inches='tight')
plt.show()



# ----------- Task 4:  White Balancing -----------

def white_balance_gray_world(img):
    img = img.astype(np.float32)
    avgR = np.mean(img[:, :, 0])
    avgG = np.mean(img[:, :, 1])
    avgB = np.mean(img[:, :, 2])

    img[:, :, 0] *= avgG / avgR  # Red
    img[:, :, 2] *= avgG / avgB  # Blue

    return np.clip(img, 0, None)

# Apply white balancing to demosaiced image
rgb_wb = white_balance_gray_world(rgb_image)

# Apply gamma for visualization
rgb_wb_gamma = gamma_correction(rgb_wb, gamma=0.3)

plt.imshow(rgb_wb_gamma)
plt.savefig(os.path.join(output_dir, "task4_white_balanced.png"), bbox_inches='tight')
plt.title("White Balanced + Gamma 0.3")
plt.axis("off")
plt.show()


# ---------- Task 5: Sensor Linearity ----------

exposure_times = [1/320, 1/160, 1/80, 1/40, 1/20, 1/10]
folder = "exercise_4_data/05"
image_paths = sorted(glob(os.path.join(folder, "*.CR3")))

avg_r, avg_g, avg_b = [], [], []

for path in image_paths:
    raw = rawpy.imread(path)
    bayer = np.array(raw.raw_image_visible, dtype=np.float32)
    R, G, B = extract_channels_RGGB(bayer)
    avg_r.append(np.mean(R[R > 0]))
    avg_g.append(np.mean(G[G > 0]))
    avg_b.append(np.mean(B[B > 0]))

exposure_times = exposure_times[::-1]  # match image order

plt.plot(exposure_times, avg_r, 'r-o', label='Red')
plt.plot(exposure_times, avg_g, 'g-o', label='Green')
plt.plot(exposure_times, avg_b, 'b-o', label='Blue')
plt.xlabel("Exposure Time (s)")
plt.ylabel("Average Pixel Value")
plt.title("Sensor Linearity Test")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "task5_sensor_linearity.png"), bbox_inches='tight')
plt.show()



# ---------- Task 6: Initial HDR Implementation ----------

# Set the directory containing 00.CR3 to 10.CR3
hdr_path = '/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/exercise_4_data/06'
hdr_files = [f"{i:02d}.CR3" for i in range(11)]
exposure_times = [1 / (2 ** i) for i in range(11)]  # t = 1.0, 0.5, ..., 1/1024

print("Processing HDR frames...")
weighted_sum = None
weight_total = None

for fname, t in zip(hdr_files, exposure_times):
    print(f"Reading {fname} with t={t}")
    raw = rawpy.imread(os.path.join(hdr_path, fname))
    bayer = np.array(raw.raw_image_visible, dtype=np.float32)
    norm_frame = bayer / t

    if weighted_sum is None:
        weighted_sum = norm_frame
        weight_total = np.ones_like(bayer)
    else:
        mask = norm_frame < 15000
        weighted_sum[mask] += norm_frame[mask]
        weight_total[mask] += 1

hdr_bayer = weighted_sum / weight_total

# Demosaicing (reusing functions)
R, G, B = extract_channels_RGGB(hdr_bayer)
R_interp = interpolate(R)
G_interp = interpolate(G)
B_interp = interpolate(B)
rgb_hdr = np.stack((R_interp, G_interp, B_interp), axis=-1)

# White balance (reuse)
rgb_hdr_wb = white_balance_gray_world(rgb_hdr)

# Log compression and percentile normalization
log_hdr = np.log10(rgb_hdr_wb + 1e-6)
log_min = np.percentile(log_hdr, 0.01)
log_max = np.percentile(log_hdr, 99.99)
log_hdr = (log_hdr - log_min) / (log_max - log_min)
log_hdr = np.clip(log_hdr, 0, 1)

# Convert to 8-bit and display
hdr_8bit = (log_hdr * 255).astype(np.uint8)
plt.imshow(hdr_8bit)
plt.savefig(os.path.join(output_dir, "task6_hdr_log_compressed.png"), bbox_inches='tight')
cv2.imwrite(os.path.join(output_dir, "task6_hdr_log_compressed.jpg"), hdr_8bit)
plt.title("HDR Output (Log Compressed)")
plt.axis("off")
plt.show()


# ---------- Task 7: iCAM06 Tone Mapping ----------

def iCAM06_tonemap(rgb_hdr, output_range=4):
    rgb_hdr = np.clip(rgb_hdr, 1e-5, None)  # Avoid log(0)

    # 1. Compute input intensity
    intensity = (20 * rgb_hdr[:, :, 0] + 40 * rgb_hdr[:, :, 1] + rgb_hdr[:, :, 2]) / 61

    # 2. Normalize RGB by intensity
    r = rgb_hdr[:, :, 0] / intensity
    g = rgb_hdr[:, :, 1] / intensity
    b = rgb_hdr[:, :, 2] / intensity

    # 3. Compute log base using a bilateral filter on log intensity
    log_intensity = np.log(intensity)
    log_base = cv2.bilateralFilter(log_intensity.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Compute detail layer
    log_detail = log_intensity - log_base

    # 5. Compute compression parameters
    compression = np.log(output_range) / (np.max(log_base) - np.min(log_base))
    log_offset = -np.max(log_base) * compression

    # 6. Output intensity
    log_output = log_base * compression + log_offset + log_detail
    output_intensity = np.exp(log_output)

    # 7. Reconstruct RGB image
    out_r = r * output_intensity
    out_g = g * output_intensity
    out_b = b * output_intensity
    output_img = np.stack([out_r, out_g, out_b], axis=-1)

    # 8. Normalize to [0, 255] for display
    output_img = np.clip(output_img / np.percentile(output_img, 99.99), 0, 1)
    output_img_uint8 = (output_img * 255).astype(np.uint8)

    return output_img_uint8

# Example:
icam_img = iCAM06_tonemap(rgb_hdr)
plt.imshow(icam_img)
plt.title("iCAM06 Tone Mapped HDR")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "task7_icam06_output.png"), bbox_inches='tight')
cv2.imwrite(os.path.join(output_dir, "task7_icam06_output.jpg"), icam_img)
plt.show()



# ---------- Task 8: process_raw function ----------

def process_raw(input_path, output_path):
    """
    Demosaics, white balances, gamma corrects and saves a high-quality JPG from a RAW CR3 file.
    """
    # Step 1: Load raw data
    raw = rawpy.imread(input_path)
    bayer = np.array(raw.raw_image_visible, dtype=np.float32)

    # Step 2: Demosaicing
    R, G, B = extract_channels_RGGB(bayer)
    R_interp = interpolate(R)
    G_interp = interpolate(G)
    B_interp = interpolate(B)
    rgb_image = np.stack((R_interp, G_interp, B_interp), axis=-1)

    # Step 3: White balance
    rgb_balanced = white_balance_gray_world(rgb_image)

    # Step 4: Gamma correction
    corrected = gamma_correction(rgb_balanced, gamma=0.3)

    # Step 5: Convert to 8-bit and save as high-quality JPG
    corrected_8bit = (corrected * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(corrected_8bit, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 99])
    print(f"✅ Saved JPG at: {output_path}")
    
    # Step 6: Display the image
    plt.imshow(corrected)
    plt.title("Task 8: Final Processed Image")
    plt.axis("off")
    plt.show()

input_cr3 = "/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/exercise_4_data/02/IMG_4782.CR3"
output_jpg = "/Users/talha/Documents/FAU 4/Project_CV/Exercise 4/results/task8_output.jpg"
process_raw(input_cr3, output_jpg)

