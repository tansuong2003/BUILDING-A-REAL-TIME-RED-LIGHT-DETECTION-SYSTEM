
# BUILDING-A-REAL-TIME-RED-LIGHT-DETECTION-SYSTEM

## 🚦 Overview

This project implements a **real-time red light violation detection system**, which initially meets key objectives in detecting and recording traffic violations at intersections.

## ✨ Features

* **Traffic Light and Vehicle Detection:**
  The system employs two YOLO models to detect traffic lights and common vehicles such as motorcycles, cars, and trucks. Through practical experiments on simulated intersection videos, the system successfully recognizes most objects under daylight conditions and moderate traffic density.

* **Data Storage and Interface:**
  Detected violations are stored directly in a **PostgreSQL database**, while a **Streamlit interface** provides an easy way to review and retrieve the recorded information.

* **Stability:**
  During trial runs, the system operated stably without crashing and was able to continuously record multiple violations.

## 🔗 Training Link

**model\ki_tu_bien_so.pt:** https://www.kaggle.com/code/thnhlong1503/train-label-bienso

**model\xeco.pt:** yolov11

**model\dengiaothong.pt:** https://www.kaggle.com/code/tnsngnguyn/train-dendo

**model\bienso.pt:** https://www.kaggle.com/code/thnhlong1503/train-bienso-2000

## 🚀 How to Run

1. **Setup environment:**

   * Configure environment variables and settings as needed.
   * Make sure Docker is installed and configured properly.

2. **Start services:**

   ```bash
   docker-compose up
   ```

3. **Run the Streamlit app:**

   ```bash
   streamlit run vuot_den_do.py
   ```

## 📂 Project Structure

```
.
├── train_model/          # Training scripts and notebooks
├── xu_ly_anh/            # Image processing modules
├── vuot_den_do.py        # Main application entry point
├── docker-compose.yml    # Docker configuration
├── README.md             # Project description
└── ...
```

## ✅ Status

🚧 This is an initial prototype that meets basic functionality goals. Future improvements may include:

* Handling low light and night-time scenarios
* Increasing detection accuracy under heavy traffic
* Adding plate recognition for automated ticketing

## 👨‍💻 Authors & Contributors
- **Nguyễn Tấn Sương** ([tansuong2003](https://github.com/tansuong2003)) - Initial project & development
- **Nguyễn Trương Thành Long** ([nguyentruongthanhlong](https://github.com/Thanhlong01052003)) - Co-developer, testing & documentation


🏷️ Tags

#yolov5 #yolov7 #yolov8 #object-detection #computer-vision #traffic-light #vehicle-detection
#streamlit #docker #docker-compose #postgresql #realtime #AI #deep-learning #pytorch
#red-light-detection #smart-traffic #smart-city #opencv #python #ALPR #license-plate-recognition
#graduation-thesis #final-project #capstone #ute #hcmute #ute-khoaluantotnghiep #ute-vietnam
#nguyentansuong #nguyentruongthanhlong #khoa-luan-tot-nghiep #hcmute-thesis #university-project
#cv #machine-learning #data-science #traffic-violation #traffic-surveillance #thesis
#iot #edge-ai #urban-traffic #motorbike-detection #car-detection #truck-detection

