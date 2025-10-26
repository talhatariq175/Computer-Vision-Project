#  Computer Vision Project – MSc Coursework

This repository contains practical exercises from my Master's degree coursework in Computer Vision at FAU Erlangen-Nürnberg.  
Each exercise explores key concepts like 3D data processing, image segmentation, and geometric analysis using Python.

---

## Contents

| Exercise | Topic                     | Description |
|----------|---------------------------|-------------|
| [Exercise 01](./exercise-01-box-detection) | Box Detection with Using RANSAC and Image Processing | Estimate a box’s dimensions using RANSAC, 3D point clouds, and image processing |
| [Exercise 02](./exercise-02-writer-retrieval) | Writer Retreival using SIFT features | Built an image retrieval system to identify document authors based on handwriting. Implemented the Bag of Visual Words (BoVW) model using VLAD encoding, k-means clustering, and Exemplar-SVMs. |

> More exercises will be added as the course progresses.

---

## Tools & Libraries Used

- Python 3
- NumPy, SciPy, Matplotlib
- OpenCV: (cv2.BFMatcher, cv2.matchTemplate)
- Scikit-learn: (e.g., MiniBatchKMeans, LinearSVC, sklearn.svm)
- Key Concepts:
  - 3D Data: 3D Point Cloud Analysis, RANSAC (custom implementation)
  - Features & ML: Bag of Visual Words (BoVW), VLAD Encoding, k-Means Clustering, k-NN Classification, SVMs
  - Deep Learning: CNNs, FaceNet Embeddings, MTCNN
  - Image Processing: Demosaicing, HDR Merging, White Balance, Morphological Operations, Template Matching
  - Object Detection: Selective Search, Region Merging



---

## Repository Structure

```text
computer-vision-project/
├── exercise-01-box-detection/
│   ├── Exercise_1_Box_Detection.ipynb
│   ├── Report_Exercise_1_Box_Detection.pdf
│   ├── Presentation_Exercise1_Box_Detection.pdf
│   └── README.md
├── exercise-02-writer-retrieval/
│   ├── Exercise_2_Writer_Identification.py
│   ├── Exercise_2_bonus.py
│   ├── Presentation_Exercise_2.pptx
│   └── README.md
├── .gitignore
└── README.md  ← you are here
