# Integrated Diabetes Risk Prediction Pipeline

**Course:** CBIO313 — Data Mining & Machine Learning  
**Author:** Malak Amr (ID: 221001703)  
**Supervisor:** Dr. Muhammad Elsayeh  

---

## Project Overview
An end-to-end machine learning project designed to predict diabetes risk early using patient clinical indicators from the Pima Indians Diabetes Dataset. The project spans exploratory data analysis, data preprocessing (imputation & scaling), baseline vs. ensemble model training, and an interactive Streamlit web application.

---

## Key Highlights & Results
* **Dataset:** 768 total patient records across 8 clinical features.
* **Best Performing Model:** **Random Forest Classifier** achieving **~78% overall accuracy** (120/154 correct test set predictions).
* **Top Clinical Predictors:** Glucose (0.49 correlation) and BMI (0.31 correlation).
* **Interactive Deployment:** Built with Streamlit (`app.py`) for real-time risk assessment via clinical sliders.

---

## Repository Contents
* `notebook.ipynb` — Data preprocessing, EDA, median imputation, scaling, and model evaluations.
* `app.py` — Source code for the interactive Streamlit clinical tool.
* `final report.pdf` — Comprehensive academic documentation.
* `Teal and White Modern Medical Diabetes Care Presentation.pptx` — Presentation slide deck.
* `requirements.txt` — Python dependencies for local and cloud deployment.
