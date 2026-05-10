from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")

def home():

    return {
        "message": "Smart Parking RL API Running"
    }


@app.get("/predict")

def predict():

    slot = random.choice(["A", "B", "C"])

    return {
        "recommended_slot": slot
    }