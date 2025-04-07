# 📝 OCR System using YOLO + CRNN + CTC Loss

## 🔍 Overview

This project is an end-to-end Optical Character Recognition (OCR) system that detects and recognizes text from images. It consists of two main stages:

1. **Text Detection** using **YOLO**
2. **Text Recognition** using **CRNN** (Convolutional Recurrent Neural Network) with **CTC Loss**

---

## 🛠️ Methodology

### 1️⃣ Text Detection with YOLO
- YOLO is used to detect and crop regions of interest (text areas) from input images.
- Each cropped region is then passed to the recognition model.

### 2️⃣ Text Recognition with CRNN
- A CRNN model with **ResNet101** backbone extracts spatial features.
- These features are passed through a **BiLSTM** to model the character sequence.
- The **CTC (Connectionist Temporal Classification)** loss is used to compute alignment-free loss for sequence-to-sequence mapping.


---

## 📈 Results (training)

![image](https://github.com/user-attachments/assets/d93bda35-87bb-4098-a67e-ee0e29035efa)


### 📸 Sample Outputs

![Untitled design](https://github.com/user-attachments/assets/7faae73f-0145-43b1-8d37-8156185eb060)



---

## 🧪 Technologies Used

- Python
- PyTorch
- YOLOv8 (for text detection)
- timm (ResNet101 backbone)
- CTC Loss
- OpenCV
- PIL

---

## 📌 Notes

- This project was developed as part of a university coursework at FPT University, majoring in Artificial Intelligence.
- Feel free to fork, contribute, or raise issues to improve this project!

---

## 👨‍🎓 Author

**Hung Anh**  
📧 Email: hunganhd1012@gmail.com


