# Exercise 04: Computational Photography – Demosaicing & HDR

This project implements a full image processing pipeline to transform raw sensor data into high-quality High Dynamic Range (HDR) images. It covers the fundamental steps of the digital camera pipeline, from Bayer pattern reconstruction to advanced tone mapping.

## Technical Pipeline

### 1. Bayer Pattern Investigation

Before processing, the sensor's Bayer layout was analyzed by inspecting raw pixel values in a 4x4 patch. By observing the distribution of high (Green/Red) and low (Blue) intensities, the pattern was identified as RGGB.

### 2. Bayer Demosaicing

We implemented a Bilinear Demosaicing algorithm to reconstruct a full RGB image from the single-channel Bayer mosaic. This involves interpolating missing color values using a weighted 3x3 convolution kernel.

### 3. White Balancing & Gamma Correction

* Gray World Algorithm: To correct color casts, we applied white balancing by scaling the Red and Blue channels based on the assumption that the average scene color is neutral gray.

* Gamma Correction: Implemented non-linear intensity transformation to convert linear sensor data into a format suitable for human perception and display.

### 4. Sensor Linearity Test

To ensure the validity of the HDR merging process, a linearity test was performed. By plotting average pixel values against exposure times (1/320s to 1/10s), we verified that the sensor responds linearly to light intensity.

### 5. High Dynamic Range (HDR) Merging

We implemented two methods for HDR creation:

* Weighted Exposure Merging: Combining multiple .CR3 raw frames using a weighting function to avoid saturated pixels and minimize noise.

* Debevec Calibration: Utilizing OpenCV's createCalibrateDebevec to estimate the Camera Response Function (CRF) and merge standard JPEGs using EXIF metadata for exposure times.

### 6. Tone Mapping Operators

To display the high-bit-depth radiance maps on 8-bit displays, we implemented multiple operators:

* Logarithmic Compression: Compressing the dynamic range using a log-base mapping.

* iCAM06 Tone Mapping: An advanced operator using bilateral filtering to separate base and detail layers, preserving local contrast while compressing the global range.

## Technologies Used

* Python 3

* rawpy: For reading raw .CR3 camera files.

* NumPy & SciPy: For matrix-based image manipulation and convolutions.

* OpenCV: For bilateral filtering, HDR calibration, and image I/O.

* Matplotlib: For linearity plots and pipeline visualization.

## Files in this Folder

* Exercise_4.py: Main implementation covering the full RAW-to-HDR pipeline (Tasks 1-8).
 
* Exercise_4_Additional.py: Implementation of HDR merging from JPEG sequences using EXIF data.

* results/: Output folder containing generated task images (Demosaiced, White Balanced, HDR).
