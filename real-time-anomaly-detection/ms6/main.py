import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from joblib import load
from prometheus_client import make_asgi_app, Counter, Histogram


class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: Optional[bool] = False


clf = load('service/model.joblib')

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

prediction_counter = Counter('prediction_counter', 'prediction counter')
model_information_counter = Counter('model_informations', 'model information counter')
prediction_histogram = Histogram('prediction_value', 'prediction histogram')
prediction_score_histogram = Histogram('prediction_score', 'prediction score histogram')
prediction_latency_histogram = Histogram('prediction_latency', 'prediction latency histogram')


@app.post("/prediction")
@prediction_latency_histogram.time()
def predict(request: PredictionRequest):
    response = {}

    prediction_counter.inc()

    prediction = clf.predict([request.feature_vector])
    pred_val = int(prediction[0])
    response["is_inlier"] = pred_val

    prediction_histogram.observe(pred_val)

    if request.score:
        scores = clf.score_samples([request.feature_vector])
        score_val = scores[0]
        response["anomaly_score"] = score_val
        prediction_score_histogram.observe(score_val)

    return response


@app.get("/model_information")
def model_information():
    model_information_counter.inc()
    return clf.get_params()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

