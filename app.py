from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.secret_key = "riddhie_secret"

# =======================
# Load Models & Encoders
# =======================
model = joblib.load("model/model.pkl")
symptom_encoder = joblib.load("model/symptom_encoder.pkl")
disease_encoder = joblib.load("model/disease_encoder.pkl")

# =======================
# Load & Normalize Datasets
# =======================
def normalize_doctors_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to a stable set that the templates expect.
    Resulting columns (lowercase): 'doctor name', 'specialization', 'hospital name', 'city', plus others preserved.
    """
    df = df.copy()
    # strip whitespace and lowercase temporarily for matching
    original_cols = list(df.columns)
    cols_lower = [c.strip().lower() for c in original_cols]

    rename_map = {}
    for orig, cl in zip(original_cols, cols_lower):
        new_name = orig.strip()  # default keep original trimmed
        # map common variants to standardized names
        if "doctor" in cl and ("name" in cl or "doctor name" in cl or "dr " in cl):
            new_name = "doctor name"
        elif "special" in cl or "specialis" in cl:  # specialization or specialisation
            new_name = "specialization"
        elif "hospital" in cl or "hospital name" in cl:
            new_name = "hospital name"
        elif cl == "city" or "city" in cl:
            new_name = "city"
        elif "address" in cl:
            new_name = "address"
        elif "phone" in cl or "contact" in cl:
            new_name = "phone"
        # else keep trimmed original (will be lowercased below)
        rename_map[orig] = new_name

    df = df.rename(columns=rename_map)
    # finally make all column names lowercase for consistent access
    df.columns = [c.strip().lower() for c in df.columns]

    # ensure required fields exist
    for required in ("doctor name", "specialization", "hospital name", "city"):
        if required not in df.columns:
            df[required] = ""  # create empty column if missing

    # fill NA
    df = df.fillna("")

    # Trim whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df

# load csv then normalize
doctors_df = pd.read_csv("doctor_location_dataset_mp.csv")
doctors_df = normalize_doctors_df(doctors_df)

precautions_df = pd.read_csv("Disease precaution.csv")
precautions_df.columns = precautions_df.columns.str.strip().str.lower()
doctors_df = doctors_df.fillna("")
precautions_df = precautions_df.fillna("")

# =======================
# Temporary in-memory storage for appointments
# =======================
appointments = []

# =======================
# Routes
# =======================
@app.route('/')
def home():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    session['user'] = {
        "name": request.form['name'],
        "phone": request.form['phone'],
        "email": request.form['email'],
        "address": request.form['address'],
        "city": request.form['city']
    }
    return redirect(url_for('symptom_page'))

@app.route('/symptom')
def symptom_page():
    return render_template("symptom.html")

@app.route('/predict', methods=['POST'])
def predict():
    raw = request.form.get('symptoms', '')
    tokens = [s.strip().lower() for s in raw.replace(';', ',').split(',') if s.strip()]
    known_symptoms = list(symptom_encoder.classes_)

    matched = [t for t in tokens if t in known_symptoms]
    unmatched = [t for t in tokens if t not in known_symptoms]

    if len(matched) == 0:
        message = ("None of the entered symptoms matched known dataset terms. "
                   "Try entering symptoms like 'fever', 'cough', 'pain', etc.")
        # show some doctors fallback (top rows)
        doctors_list = doctors_df.head(10).to_dict(orient='records')
        return render_template("predict.html",
                               disease="No reliable prediction",
                               precaution=message,
                               matched=matched,
                               unmatched=unmatched,
                               doctors=doctors_list,
                               no_city_msg="")

    # Predict disease
    X_input = symptom_encoder.transform([matched])
    pred = model.predict(X_input)
    disease = disease_encoder.inverse_transform(pred)[0]

    # Get precautions (case-insensitive match)
    info = precautions_df[precautions_df["disease"].str.lower() == disease.lower()]
    precaution = "No data available."
    if not info.empty:
        precaution_cols = [c for c in info.columns if "precauti" in c.lower()]
        precaution_list = [str(info.iloc[0][c]).strip() for c in precaution_cols if pd.notna(info.iloc[0][c])]
        precaution = ", ".join(precaution_list) if precaution_list else precaution

    # City and specialization matching
    user_city = session.get('user', {}).get('city', '').strip().lower()

    specialization_map = {
        "fever": "general physician",
        "infection": "general physician",
        "cold": "general physician",
        "flu": "general physician",
        "heart": "cardiologist",
        "cardio": "cardiologist",
        "diabetes": "endocrinologist",
        "thyroid": "endocrinologist",
        "skin": "dermatologist",
        "rash": "dermatologist",
        "acne": "dermatologist",
        "bone": "orthopedic",
        "fracture": "orthopedic",
        "tooth": "dentist",
        "gum": "dentist",
        "eye": "ophthalmologist",
        "vision": "ophthalmologist",
        "depression": "psychiatrist",
        "anxiety": "psychiatrist",
        "pregnancy": "gynecologist",
        "child": "pediatrician"
    }

    disease_lower = disease.lower()
    recommended_specialization = "general physician"
    for keyword, spec in specialization_map.items():
        if keyword in disease_lower:
            recommended_specialization = spec
            break

    # Ensure columns exist and compare in lowercase
    doctors_df['specialization'] = doctors_df['specialization'].astype(str)
    doctors_df['city'] = doctors_df['city'].astype(str)

    available_docs = pd.DataFrame()
    if user_city:
        mask_city = doctors_df['city'].str.lower() == user_city
        mask_spec = doctors_df['specialization'].str.lower().str.contains(recommended_specialization.lower(), na=False)
        available_docs = doctors_df[mask_city & mask_spec]
    # fallback: specialization only
    if available_docs.empty:
        available_docs = doctors_df[doctors_df['specialization'].str.lower().str.contains(recommended_specialization.lower(), na=False)]
        no_city_msg = f"No {recommended_specialization.title()}s found in your city — showing nearby doctors."
    else:
        no_city_msg = ""

    doctors_list = available_docs.head(10).to_dict(orient='records')

    # ✅ Store prediction data in session for report
    session["symptoms"] = raw
    session["predicted_disease"] = disease
    session["precautions"] = precaution

    return render_template("predict.html",
                           disease=disease,
                           precaution=precaution,
                           matched=matched,
                           unmatched=unmatched,
                           doctors=doctors_list,
                           no_city_msg=no_city_msg)

# ✅ Book Appointment
@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    doctor_name = request.form.get('doctor_name')
    specialization = request.form.get('doctor_specialization')
    city = request.form.get('city')

    user = session.get('user', {})

    random_days = random.randint(1, 7)
    random_hour = random.choice(range(9, 17))
    random_minute = random.choice([0, 15, 30, 45])
    appointment_datetime = datetime.now() + timedelta(days=random_days)
    appointment_datetime = appointment_datetime.replace(hour=random_hour, minute=random_minute, second=0)

    formatted_date = appointment_datetime.strftime("%B %d, %Y")
    formatted_time = appointment_datetime.strftime("%I:%M %p")

    appointment = {
        "name": user.get("name"),
        "phone": user.get("phone"),
        "email": user.get("email"),
        "city": city,
        "doctor_name": doctor_name,
        "doctor_specialization": specialization,
        "appointment_date": formatted_date,
        "appointment_time": formatted_time
    }
    appointments.append(appointment)

    return render_template("appointment.html", data=appointment)

# ✅ Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html", appointments=appointments)

# ✅ Trends
@app.route('/trends')
def trends():
    if not appointments:
        return render_template("trends.html",
                               city_counts={},
                               specialization_counts={},
                               message="No appointment data available yet. Book some appointments to see trends!")

    df = pd.DataFrame(appointments)
    city_counts = df['city'].value_counts().to_dict()
    specialization_counts = df['doctor_specialization'].value_counts().to_dict()

    return render_template("trends.html",
                           city_counts=city_counts,
                           specialization_counts=specialization_counts,
                           message="")

# ✅ Doctors Page
@app.route('/doctors')
def doctors_page():
    user = session.get('user', {})
    city = user.get('city', '').strip().lower()

    # doctors_df already normalized at load time
    if city and city in doctors_df['city'].str.lower().values:
        city_doctors = doctors_df[doctors_df['city'].str.lower() == city]
        msg = f"Doctors available in {city.title()}"
    else:
        city_doctors = doctors_df.head(10)
        msg = "No doctors found in your city. Showing nearby doctors in MP."

    return render_template("doctors.html",
                           city=city or "MP",
                           doctors=city_doctors.to_dict(orient='records'),
                           message=msg)

# ✅ Report Page (renders report.html)
@app.route("/report")
def report_page():
    return render_template("report.html")

# ✅ Report Generation (Enhanced PDF with Dashboard + Trends + Appointments)
@app.route("/generate_report", methods=["GET", "POST"])
def generate_report():
    user = session.get('user', {})
    user_name = user.get("name", "Guest")
    user_phone = user.get("phone", "Not provided")
    user_email = user.get("email", "Not provided")
    user_city = user.get("city", "Unknown")
    user_address = user.get("address", "Not provided")

    predicted_disease = session.get("predicted_disease", "Not diagnosed")
    symptoms = session.get("symptoms", "Not provided")
    precautions = session.get("precautions", "No precautions available")

    # Prepare trend data if available
    if appointments:
        df = pd.DataFrame(appointments)
        city_counts = df['city'].value_counts().to_dict()
        specialization_counts = df['doctor_specialization'].value_counts().to_dict()
    else:
        city_counts = {}
        specialization_counts = {}

    # ===== Create Graphs for Trends =====
    trend_images = []

    if city_counts:
        plt.figure(figsize=(5, 3))
        plt.bar(city_counts.keys(), city_counts.values(), color='#4CAF50')
        plt.title("City-wise Appointments Trend")
        plt.xlabel("City")
        plt.ylabel("Number of Appointments")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig("city_trend.png")
        plt.close()
        trend_images.append("city_trend.png")

    if specialization_counts:
        plt.figure(figsize=(5, 3))
        plt.bar(specialization_counts.keys(), specialization_counts.values(), color='#2196F3')
        plt.title("Specialization-wise Appointments Trend")
        plt.xlabel("Specialization")
        plt.ylabel("Number of Appointments")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig("specialization_trend.png")
        plt.close()
        trend_images.append("specialization_trend.png")

    # ===== Generate PDF =====
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)

    def safe_text(text):
        # Remove unsupported characters (like emojis)
        return text.encode('latin-1', 'ignore').decode('latin-1')

    # ---------------- Header ----------------
    pdf.cell(200, 10, txt=safe_text("AI Health & Symptom Report"), ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text(f"Name: {user_name}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"City: {user_city}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"Email: {user_email}"), ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=safe_text(f"Entered Symptoms: {symptoms}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"Predicted Disease: {predicted_disease}"), ln=True)
    pdf.multi_cell(0, 8, txt=safe_text(f"Precautions: {precautions}"))
    pdf.ln(8)

    # ---------------- Dashboard Summary ----------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=safe_text("Dashboard Summary"), ln=True)
    pdf.set_font("Arial", "", 12)
    if appointments:
        pdf.cell(200, 10, txt=safe_text(f"Total Appointments: {len(appointments)}"), ln=True)
    else:
        pdf.cell(200, 10, txt=safe_text("No appointment data available."), ln=True)
    pdf.ln(5)

    # ---------------- All Booked Doctor Appointments ----------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=safe_text("All Booked Doctor Appointments:"), ln=True)
    pdf.set_font("Arial", "", 12)

    booked_doctors = []
    try:
        booked_df = pd.read_csv("BookedDoctors.csv")  # Your booking record file
        booked_doctors = booked_df.to_dict(orient="records")
    except Exception:
        booked_doctors = []

    if booked_doctors:
        for doc in booked_doctors:
            pdf.multi_cell(
                0, 8,
                txt=safe_text(
                    f"Doctor Name: {doc.get('DoctorName', 'N/A')}\n"
                    f"Speciality: {doc.get('Speciality', 'N/A')}\n"
                    f"Booking Date: {doc.get('BookingDate', 'N/A')}\n"
                    "-----------------------------------"
                )
            )
    else:
        pdf.cell(200, 10, txt=safe_text("No booked doctor appointments found."), ln=True)

    pdf.ln(10)

    # ---------------- Trends Overview ----------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=safe_text("Trends Overview"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt=safe_text("These graphs show current health trends across regions and specialities."), ln=True)

    # ✅ Add saved trend graphs into PDF
    for img_path in trend_images:
        try:
            pdf.image(img_path, x=30, w=150)
            pdf.ln(5)
        except:
            pdf.cell(200, 10, txt=safe_text(f"Unable to add graph: {img_path}"), ln=True)

    # ---------------- Save Report ----------------
    pdf.output("Health_Report.pdf")
    return send_file("Health_Report.pdf", as_attachment=True)


# =======================
# Run Flask App
# =======================
if __name__ == "__main__":
    app.run(debug=True)
