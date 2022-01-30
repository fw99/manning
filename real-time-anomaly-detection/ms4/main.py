import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from joblib import load


class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: Optional[bool] = False


clf = load('service/model.joblib')

app = FastAPI()


@app.post("/prediction")
def predict(request: PredictionRequest):
    response = {}

    prediction = clf.predict([request.feature_vector])
    response["is_inlier"] = int(prediction[0])

    if request.score:
        scores = clf.score_samples([request.feature_vector])
        response["anomaly_score"] = scores[0]

    return response


@app.get("/model_information")
def model_information():
    return clf.get_params()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
