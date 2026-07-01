from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Salary Prediction API")

# Load ML Model
model = joblib.load("salary_model.pkl")


class SalaryInput(BaseModel):
    experience: float


@app.get("/")
def home():
    return {"message": "Salary Prediction API Running"}


@app.post("/predict")
def predict(data: SalaryInput):

    input_data = pd.DataFrame({
        "YearsExperience": [data.experience]
    })

    prediction = model.predict(input_data)

    return {
        "experience": data.experience,
        "predicted_salary": float(prediction[0])
    }