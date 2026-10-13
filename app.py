import csv
import os
from datetime import datetime
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

model = joblib.load("model.pkl")
MODEL_METRICS = {
    "mae": 0.71,
    "rmse": 0.81,
    "r2": 0.9911
}
@app.get("/metrics")
def get_metrics():
    return MODEL_METRICS
from pathlib import Path

HISTORY_FILE = Path(__file__).resolve().parent / "prediction_history.csv"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "study_hours",
            "sleep_hours",
            "attendance",
            "previous_score",
            "predicted_final_score"
        ])

from pydantic import BaseModel, Field
class StudentData(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    sleep_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)


@app.post("/predict")
def predict_final_score(data: StudentData):
    student = pd.DataFrame(
        [[data.study_hours, data.sleep_hours, data.attendance, data.previous_score]],
        columns=["study_hours", "sleep_hours", "attendance", "previous_score"],
    )
    prediction = model.predict(student)
    predicted_score = round(float(prediction[0]), 2)

    with open(HISTORY_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.study_hours,
            data.sleep_hours,
            data.attendance,
            data.previous_score,
            predicted_score,
        ])

    return {"predicted_final_score": predicted_score}
@app.get("/history")
def get_prediction_history():
    with open(HISTORY_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)
@app.delete("/history")
def clear_prediction_history():
    with open(HISTORY_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "study_hours",
            "sleep_hours",
            "attendance",
            "previous_score",
            "predicted_final_score"
        ])

    return {"message": "Prediction history cleared successfully"}