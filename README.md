# 🧠 AI Symptom Checker and Doctor Recommender System

## 📘 Project Overview

AI Symptom Checker and Doctor Recommender is an intelligent healthcare web application that predicts diseases based on user-input symptoms and recommends nearby doctors according to the user’s city and disease specialization.

This Flask-based web app uses Machine Learning models and medical datasets to provide disease predictions, preventive measures, doctor listings, appointment booking, and downloadable health reports — offering a smart digital healthcare experience.

## 🎯 Objective

To design a smart health assistant that:

Predicts diseases from given symptoms using AI.

Recommends nearby doctors based on specialization and city.

Provides precautionary healthcare tips and cost insights.

Allows booking, managing, and viewing past appointments.

Generates and downloads personal health reports.

## 🚀 Key Features
Category	Description
🧾 Symptom Checker	Users enter symptoms and get disease predictions using an ML model.
🧠 AI Disease Prediction	Uses trained Random Forest model for multi-symptom analysis.
👨‍⚕️ Doctor Recommendation	Recommends doctors based on predicted disease and selected city.
🏥 Available Doctors by City	Lists all doctors and specializations available in the chosen city.
💬 Precaution Tips	Displays 4 key precautionary measures for the predicted disease.
📅 Book Appointment	Allows booking with a selected doctor; data stored in appointments.db.
📋 Dashboard	Displays all current and past appointments for the logged-in user.
📊 Trend Visualization	Shows disease and specialization trends using graphs and charts.
📄 Health Report Generation	Creates a downloadable PDF report summarizing prediction and doctor info.
🔒 User Login System	Simple login form with name and email for session management.

## 📂 Project Structure
AI_Symptom_Checker_Doctor_Recommender/
│
├── app.py                           # Main Flask web app
├── model_train.py                   # Machine Learning model training
├── appointments.db                  # Database for appointment storage
├── DiseaseAndSymptoms.csv           # Dataset for symptom-disease mapping
├── Disease precaution.csv           # Dataset with precautions for each disease
├── doctor_location_dataset_mp.csv   # Dataset containing doctors and locations
├── Health_Report.pdf                # Sample health report
├── requirements.txt                 # Dependencies list
└── static/ & templates/             # Frontend UI files

## 🧰 Technologies Used
Component	Technology
Framework	Flask
Machine Learning	Scikit-learn, Pandas, Numpy
Frontend	HTML, CSS, JavaScript
Database	SQLite (appointments.db)
Visualization	Matplotlib / Seaborn
Report Generation	ReportLab / FPDF
Data Files	CSV datasets

## 🩺 Datasets Used
Dataset Name	Features	Records	Purpose
DiseaseAndSymptoms.csv	18	4920	Maps diseases with multiple symptoms
Disease precaution.csv	5	41	Stores disease-specific precautionary tips
doctor_location_dataset_mp.csv	10	500	Contains doctor details, specialization, and location

Total Features: 33

## ⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/YourUsername/AI-Symptom-Checker-Doctor-Recommender.git
cd AI-Symptom-Checker-Doctor-Recommender

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate  # For macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py

5️⃣ Access the App in Browser
http://127.0.0.1:5000/

## 🧮 Workflow Summary

Login Page → Enter name & email to start.

Symptom Entry → Input symptoms for prediction.

Disease Prediction → ML model predicts possible diseases.

Precaution Display → System shows relevant care tips.

Doctor Recommendation → View doctors by specialization & city.

Available Doctors Section → Lists all doctors of different types in the selected city.

Book Appointment → Choose doctor & book an appointment.

Dashboard → View booked, ongoing, and past appointments.

Trend Visualization → Graphs show top specializations & disease trends.

Download Report → Get PDF summary of all details.

## 📸 Screenshots

### 🏠 Loginpage
<img width="1920" height="987" alt="Screenshot (136)" src="https://github.com/user-attachments/assets/f0796d03-6bbb-4152-9f62-dc8133ab7b8f" />


### 🤒 Symptom Input
<img width="1920" height="993" alt="Screenshot (137)" src="https://github.com/user-attachments/assets/8968f0ff-0b7c-46fc-9a35-28634c2998ee" />


### 🧠 Disease Prediction Result
<img width="1920" height="990" alt="Screenshot (138)" src="https://github.com/user-attachments/assets/e3a660f5-2876-47f2-9b51-ac127f8529dc" />


### 👨‍⚕️ Doctor Recommendation
<img width="617" height="396" alt="Screenshot (138)" src="https://github.com/user-attachments/assets/2a9efc7d-6a35-4803-85de-d1c67d706139" />

### 📅 Book Appointment 
<img width="1920" height="990" alt="Screenshot (139)" src="https://github.com/user-attachments/assets/8277b737-7fd2-431b-91e8-64b0a7ddd1b1" />

<img width="1920" height="982" alt="Screenshot (139)" src="https://github.com/user-attachments/assets/add5f0d6-8256-451d-a7ad-9a213e755523" />

### 📊 Trend Visualization
<img width="1920" height="990" alt="Screenshot (141)" src="https://github.com/user-attachments/assets/6ccabdd6-2ed6-4d06-aa51-954bff9d157f" />

### 📋 Dashboard (Appointments Overview)
<img width="1920" height="987" alt="Screenshot (140)" src="https://github.com/user-attachments/assets/1310574c-0544-40af-9b1d-a362cafb423a" />

### 🏥 Available Doctors in Selected City
<img width="1920" height="973" alt="Screenshot (142)" src="https://github.com/user-attachments/assets/97d5425a-216f-447c-8542-ba804b29ccf0" />

### 📄 Health Report
<img width="1920" height="979" alt="Screenshot (143)" src="https://github.com/user-attachments/assets/c8108910-3fcf-4737-a0a9-6a7caea066e8" />

<img width="1764" height="1137" alt="Screenshot (145)" src="https://github.com/user-attachments/assets/516be8c6-9fee-4ae8-a2f4-bef2edbf22f9" />

<img width="1455" height="1030" alt="Screenshot (148)" src="https://github.com/user-attachments/assets/51414b0b-2e5b-4c3b-9cac-bc5506b39874" />

<img width="1448" height="1035" alt="Screenshot (149)" src="https://github.com/user-attachments/assets/8362cedb-1ae7-4071-a618-9b33552809d8" />


## 📊 Model Training

The ML model uses the DiseaseAndSymptoms dataset to learn symptom–disease patterns.

Algorithm: Random Forest Classifier

Input: Symptoms (Symptom_1 – Symptom_17)

Output: Predicted disease name

Model performance can be improved by data cleaning and feature engineering.

##👩‍💻 Developed By

Riddhie Sengar
Department of Mathematics and Computing
📍 3rd Year Student

## 🌟 Future Enhancements

Integration with live doctor APIs (e.g., Practo).

Real-time geolocation mapping (Google Maps API).

Chatbot for health guidance.

Mobile app version using Flutter or React Native.

Generate Report → downloadable PDF summary generated automatically.

Trends Dashboard → view disease and doctor specialization trends.
