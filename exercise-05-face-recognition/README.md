# Exercise 05: Face Recognition & Re-Identification System

This project implements a complete, end-to-end face recognition system capable of both supervised identification (recognizing known people) and unsupervised re-identification (clustering unknown people). It also addresses the "Open-Set Recognition" problem, where the system must identify if a person is "unknown" to the gallery.

 ## Technical Pipeline

### 1. Detection & Tracking

* Detection: We utilized MTCNN (Multi-task Cascaded Convolutional Networks) for robust face detection and landmark localization.

* Tracking: To improve efficiency in video streams, we implemented Template Matching to track faces between frames, significantly reducing the need for expensive MTCNN inference on every frame.

### 2. Feature Extraction (FaceNet)

We used a pre-trained FaceNet (ResNet50) model to map face images into a 128-dimensional embedding space. The model is trained so that squared L2 distances between embeddings correspond to face similarity.

* Normalization: L2 normalization is applied to embeddings to ensure consistency.

* Robustness: The system extracts embeddings for both color and grayscale versions of faces to improve recognition across different lighting conditions.

### 3. Identification & Re-ID

* Supervised Identification: Implemented a k-Nearest Neighbors (k-NN) classifier. We used a weighted voting scheme based on both posterior probability and distance thresholds to handle open-set scenarios (detecting "unknown" individuals).

* Unsupervised Re-ID: Implemented k-Means Clustering from scratch to group multiple sightings of the same unknown person without prior labels.

### 4. Open-Set Recognition (OSR)

We implemented and evaluated two specific approaches for handling unknown classes in a machine learning context:

* Single Pseudo Label (SPL): Treating all unknown samples as a single "background" class.

* Multi Pseudo Label (MPL): Assigning unique pseudo-labels to each unknown sample to better capture the variance of the "unknown" space.

### 5. Evaluation

The system's performance is evaluated using DIR (Detection and Identification Rate) curves at various False Alarm Rates (FAR), analyzing the trade-off between correctly identifying known subjects and wrongly identifying unknown ones.

## Technologies Used

* Python 3

* TensorFlow / MTCNN: For deep-learning-based face detection.

* OpenCV: For image processing, template matching, and DNN module (FaceNet inference).

* Scikit-learn: For SVMs (OSR training) and evaluation metrics.

* NumPy: For vector operations and custom k-Means implementation.

## Files in this Folder

* face_detector.py: Face detection (MTCNN) and tracking (Template Matching) logic.

* face_recognition.py: FaceNet embedding extraction, k-NN identification, and k-Means clustering.

* osr_learning.py: Implementations of SPL and MPL for open-set recognition.

* evaluation.py: DIR curve and threshold selection logic.

* training.py & test.py: Command-line interfaces for training the gallery and running real-time inference.
