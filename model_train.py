import pandas as pd
import joblib, os, re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def normalize_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = s.replace(';', ',')
    # keep commas and words+spaces; remove other punctuation
    s = re.sub(r'[^a-z0-9, ]+', ' ', s)
    # collapse spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # ensure commas separated
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return " ".join(parts)   # vectorizer sees a single string

# Load dataset
df = pd.read_csv("DiseaseAndSymptoms.csv")

# If dataset has multiple Symptom columns, combine them:
symptom_cols = [c for c in df.columns if "symptom" in c.lower()]
if len(symptom_cols) > 1:
    df["Symptoms_combined"] = df[symptom_cols].astype(str).apply(lambda row: ','.join([x for x in row if x!='nan']), axis=1)
    df["sym_text"] = df["Symptoms_combined"].apply(normalize_text)
elif "symptoms" in [c.lower() for c in df.columns]:
    col = [c for c in df.columns if c.lower()=="symptoms"][0]
    df["sym_text"] = df[col].apply(normalize_text)
else:
    raise SystemExit("No Symptoms column found - please check column names")

# drop rows with empty sym_text or disease
df = df.dropna(subset=["sym_text", "Disease"])
df = df[df["sym_text"].str.strip() != ""]

# labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"].astype(str))

# pipeline
pipeline = Pipeline([
    ("vect", CountVectorizer(ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")),
    ("clf", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))
])

# train/test split for quick eval
X = df["sym_text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)
print("Test accuracy:", pipeline.score(X_test, y_test))

# train on full for production
pipeline.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model_pipeline.pkl")
joblib.dump(le, "model/label_encoder.pkl")
print("Saved model_pipeline.pkl and label_encoder.pkl in model/")
