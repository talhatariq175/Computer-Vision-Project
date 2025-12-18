#  Computer Vision Project – MSc Coursework

This repository contains practical exercises from my Master's degree coursework in Computer Vision at FAU Erlangen-Nürnberg. Each exercise explores key concepts in 3D data processing, image segmentation, feature extraction, and deep learning using Python.

---

## Contents

| Exercise | Topic                     | Description |
|----------|---------------------------|-------------|
| [Exercise 01](./exercise-01-box-detection) | Box Detection with Using RANSAC and Image Processing | Estimate a box’s dimensions using RANSAC, 3D point clouds, and image processing |
| [Exercise 02](./exercise-02-writer-retrieval) | Writer Retreival using SIFT features | Built an image retrieval system to identify document authors based on handwriting. Implemented the Bag of Visual Words (BoVW) model using VLAD encoding, k-means clustering, and Exemplar-SVMs |
| [Exercise 03](./exercise-03-selective-search) | Selective Search for Object Detection | Implemented the Selective Search algorithm from scratch. Used Felzenszwalb's segmentation and hierarchical region merging based on color, texture, and size similarity |
| [Exercise 04](./exercise-04-demosaicing-hdr) | Demosaicing & High Dynamic Range (HDR) | Developed a full pipeline for processing raw sensor data. Implemented demosaicing from Bayer patterns, white balancing (Gray World), and merging multiple exposures to create an HDR image |
| [Exercise 05](./exercise-05-face-recognition) | Face Recognition System | Built a complete face recognition and re-identification system. Used MTCNN for detection, FaceNet for deep feature extraction (embeddings), and k-NN / k-Means for identification and clustering |


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
Computer-Vision-Project/
├── exercise-01-box-detection/
│   ├── Exercise_1_Box_Detection.ipynb
│   ├── Report_Exercise_1_Box_Detection.pdf
│   └── README.md
├── exercise-02-writer-retrieval/
│   ├── Exercise_2_Writer_Identification.py
│   ├── Exercise_2_bonus.py
│   ├── Presentation_Exercise_2.pptx
│   └── README.md
├── exercise-03-selective-search/
│   ├── baloon_detector.py
│   ├── writeup_exercise3.2.pdf
│   └── README.md
├── exercise-04-demosaicing-hdr/
│   ├── Exercise_4.py
│   ├── Exercise_4_Additional.py
│   ├── Report_Exercise_4.pdf
│   └── README.md
├── exercise-05-face-recognition/
│   ├── face_detector.py
│   ├── face_recognition.py
│   ├── osr_learning.py
│   ├── evaluation.py
│   ├── training.py
│   ├── test.py
│   └── README.md
├── .gitignore
└── README.md  ← you are here
