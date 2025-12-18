# Exercise 03: Selective Search & Balloon Detection Pipeline

This project implements a complete object detection pipeline to detect balloons in images. It combines a classic region proposal algorithm with a modern deep learning feature extractor and a machine learning classifier.

 ## Pipeline Methodology

### 1. Region Proposals: Selective Search

We implemented a simplified version of the Selective Search algorithm. The process begins with Felzenszwalb's graph-based segmentation to over-segment the image. We then extract bounding boxes from these segments to serve as candidate regions (proposals) where objects might be located.

### 2. Dataset Preparation (Proposal Matching)

To train the classifier, we match generated proposals against ground-truth balloon annotations using the Intersection over Union (IoU) metric:

* Positive Samples: Proposals with an IoU > 0.5 with a ground-truth box.

* Negative Samples: Proposals with an IoU < 0.2 (considered background).

### 3. Feature Extraction: ResNet18

For each proposed region, we extract high-level visual features using a pre-trained ResNet18 model. The regions are warped to 224x224 pixels and passed through the CNN to obtain a 512-dimensional feature vector.

### 4. Classification: SVM

A Support Vector Machine (SVM) classifier is trained on these feature vectors to distinguish between "balloon" and "background." During inference, the classifier provides a probability score for each proposal.

## Results

The pipeline successfully detects balloons in test images. Final detections are filtered using a probability threshold (0.4) and visualized with green bounding boxes.

## Technologies Used

* Python 3

* PyTorch (torchvision): ResNet18 feature extraction.

* Scikit-learn: SVM classifier and data shuffling.

* Scikit-image: Felzenszwalb segmentation and image processing.

* Matplotlib: Result visualization.

## Files in this Folder

* baloon_detector.py: The complete detection pipeline script.

* writeup_exercise3.2.pdf: Detailed technical report and analysis.

* output/: Folder containing images with predicted detections.
