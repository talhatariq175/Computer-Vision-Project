# 📦 Exercise 01 – Box Detection using RANSAC and Image Processing

This exercise is part of my Master's coursework in Computer Vision Project.  
The task was to estimate the dimensions of a box using Time-of-Flight (ToF) sensor data through RANSAC and Image Processing.

---

## 🧠 Project Summary

We implemented a pipeline to detect a box placed on the floor and estimate its **height, length, and width**.  
The process involves:

- Loading and visualizing ToF amplitude, distance, and point cloud data
- Applying a custom **RANSAC** algorithm to detect planar surfaces (floor and box top)
- Using **morphological operations** to clean segmentation masks
- Extracting geometric features from the 3D point cloud

---

## 🔍 Techniques Used

- Python with NumPy, Matplotlib, SciPy
- Custom RANSAC for plane fitting
- 3D point cloud filtering
- Morphological filtering (opening & closing)
- Connected component analysis for top surface selection

---

## 📂 Files Included

| File | Description |
|------|-------------|
| `Exercise_1_Box_Detection.ipynb` | Main notebook containing the full pipeline |
| `Report_Exercise_1_Box_Detection.pdf` | Formal project report |
| `Presentation_Exercise1_Box_Detection.pdf` | Slides used for in-class presentation |

---

## 🗂️ Dataset

> **Note**: `.mat` files used in this project are not uploaded due to size limits.  
You can place them in a local `data/` folder or request them if needed.

---

## 👥 Authors

- Talha Tariq  
- Muhammad Tahir Mubeen  
> FAU Erlangen-Nürnberg, MSc Artificial Intelligence

---

## 🔮 Future Work

- Add filtering (e.g., median filter) to reduce noise before RANSAC
- Automatically tune thresholds based on scene statistics
- Extend the pipeline to detect multiple boxes

---

## 📌 License

This project is for educational use only.
