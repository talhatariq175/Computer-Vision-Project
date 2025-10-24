# Exercise 02: Writer Identification

This project implements a system to identify the author of a historical document by analyzing their handwriting. The system is based on the Bag of Visual Words (BoVW) model, using local SIFT features to create a global "fingerprint" for each document. The goal is to match a query document to a database of known writers, evaluated using the mean Average Precision (mAP) metric.

The core pipeline follows the VLAD (Vector of Locally Aggregated Descriptors) encoding method, which is then refined using Exemplar-SVMs.

## Core Pipeline & Methodology

### 1. Codebook Generation

A "vocabulary" of visual words was created by sampling 500,000 local SIFT descriptors from the training set. These descriptors were then clustered into 100 centers using scikit-learn's MiniBatchKMeans. These 100 cluster centers form the "codebook."

### 2. VLAD Encoding

For each image, every local descriptor is assigned to its nearest cluster center in the codebook (using cv2.BFMatcher). The residual (the difference vector between the descriptor and its center) is calculated. The residuals for all descriptors belonging to the same cluster are summed up. Finally, these summed vectors are concatenated into one large VLAD vector that represents the entire image.

### 3. VLAD Normalization

To improve retrieval performance, two normalization steps are applied:

* Power Normalization: A signed square-root (np.sign(v) * np.sqrt(np.abs(v))) is applied to each element to reduce the "burstiness" of frequent features.

* $L_2$ Normalization: The final vector is normalized to unit length, ensuring that all image "fingerprints" can be compared fairly.

### 4. Exemplar-SVM Enhancement

To make the VLAD encoding more discriminative, each test image's VLAD vector is used as a positive example to train a new LinearSVC (Exemplar-SVM). All training set VLAD vectors are used as negative examples. The resulting SVM's weight vector is extracted and L2-normalized. This new vector serves as the final, highly discriminative descriptor for the test image.

## 📈 Results

The system's performance was measured by its mean Average Precision (mAP) in retrieving all correct documents from the database.

* Baseline VLAD mAP: 0.739

* Exemplar-SVM mAP: 0.825

This demonstrates a significant 8.6% improvement in retrieval accuracy by refining the VLAD vectors with a discriminative classifier.

## Bonus: Custom RootSIFT Feature Extraction

This project also includes a bonus solution (Exercise_2_Bonus.py) that implements a custom feature extractor instead of using the pre-computed features. This involved:

1. Detecting and computing SIFT features using cv2.SIFT_create().

2. Setting all keypoint angles to 0.

3. Applying Hellinger Normalization (RootSIFT) to the descriptors (L1 norm followed by element-wise signed square root).

## 🛠️ Technologies & Libraries Used

* Python 3

* Scikit-learn: MiniBatchKMeans, LinearSVC

* OpenCV: cv2.BFMatcher (for nearest neighbor search), cv2.SIFT_create

* NumPy: For all numerical operations and vector manipulation.

* Parmap: For parallel processing to speed up encoding of the dataset.

## 📂 Files in this Folder

Exercise_2_Writer_Identification.py: The main solution implementing the VLAD and Exemplar-SVM pipeline.

Exercise_2_Bonus.py: The bonus solution that includes the custom RootSIFT feature extractor.

Presentation_Exercise_2.pptx: A slide deck summarizing the project's methodology and results.
