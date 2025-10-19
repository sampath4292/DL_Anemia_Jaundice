# 🩺 Deep Learning for Anemia & Jaundice Detection from Eye Images

A web-based, non-invasive screening tool that uses deep learning to detect signs of **Anemia** and **Jaundice** from digital images of the eye.  

This project leverages **Convolutional Neural Networks (CNNs)** to provide a rapid, accessible, and low-cost preliminary health assessment.  

This repository contains the source code for the research paper:  
*"Deep Learning-Based Eye Image Analysis for Detection of Anemia and Jaundice".*

---

## 🌟 Overview

Traditional methods for diagnosing Anemia and Jaundice require invasive blood tests, which can be costly and inaccessible in remote areas.  

This project addresses this challenge by creating an AI-powered system that analyzes key biomarkers in the eye:

- **Anemia:** Detected by analyzing the pallor of the **palpebral conjunctiva** (the inner lining of the eyelid).  
- **Jaundice:** Detected by analyzing the yellowing (**icterus**) of the **sclera** (the white of the eye).  

The system integrates two separate deep learning models into a single, user-friendly web interface built with **Streamlit**, allowing for real-time predictions from a simple image upload.

---

## ✨ Key Features

- **Dual-Disease Detection:** A unified platform to screen for both Anemia and Jaundice.  
- **Non-Invasive:** Requires only a digital photograph of the eye, eliminating the need for blood draws.  
- **Web-Based Interface:** Simple and intuitive UI for easy access and real-time predictions.  
- **High Accuracy:** Built on the MobileNetV2 architecture using transfer learning for robust performance.  
- **Accessible Healthcare:** Aims to provide preliminary health screening in underserved and remote communities.  

---

## 📸 Screenshots

*(Replace these placeholders with actual screenshots of your application)*

- Home Page  
- Anemia Prediction  
- Jaundice Prediction  

---

## 💻 Technology Stack

- **Backend & Machine Learning:** Python, TensorFlow, Keras  
- **Image Processing:** OpenCV  
- **Web Framework:** Streamlit  
- **Data Handling:** NumPy, Pandas  

---

## 🔧 System Architecture

The project employs two distinct CNN models, one for each condition. The workflow is as follows:

1. **Input:** The user selects a condition (Anemia or Jaundice) and uploads a corresponding eye image (conjunctiva or sclera).  
2. **Preprocessing:** The image is resized to 224x224 pixels and normalized.  
3. **Prediction:** The image is passed to the appropriate pre-trained MobileNetV2-based model.  
4. **Output:** The model returns a binary classification (e.g., "Anemia Detected" or "No Anemia") along with a confidence score.  

*(Placeholder for the architecture diagram from your paper)*

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.8+  
- pip and venv  

### Installation & Setup

Clone the repository:

```bash
git clone https://github.com/sampath4292/DL_Anemia_Jaundice.git
cd DL_Anemia_Jaundice

Create and activate a virtual environment:

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate


On Windows:

python -m venv venv
.\venv\Scripts\activate


Install the required dependencies:

pip install -r requirements.txt

```
Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt in your project's activated virtual environment.

Run the Streamlit application:

streamlit run app.py


The application should now be running and accessible in your web browser at:
http://localhost:8501

📊 Model Performance

The models were evaluated on an unseen test set, achieving the following performance:

Model	Accuracy (%)	Precision (%)	Recall (%)	F1-Score (%)
Anemia Detection	95.2	96.1	94.5	95.3
Jaundice Detection	96.5	97.2	95.8	96.5

Disclaimer: These results are from a controlled dataset. Real-world performance may vary.

📈 Future Work

Dataset Expansion: Incorporating more diverse data across different ethnicities and lighting conditions.

Quantitative Prediction: Moving from classification to regression to estimate actual hemoglobin or bilirubin levels.

Mobile Application: Developing a native mobile app for even greater accessibility.

Clinical Validation: Conducting rigorous trials to validate the tool's performance against standard clinical diagnostics.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments

This project was completed under the esteemed guidance of Mrs. N. Kavitha, Assistant Professor, Department of CSE, Vignan's Institute of Information Technology.
