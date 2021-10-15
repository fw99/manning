```bash
curl -X 'POST' \
  'http://localhost:8000/prediction' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "feature_vector": [
    -0.157953,-0.106749
  ],
  "score": false
}'
```

{
  "is_inlier": 1
}

```bash
curl -X 'POST' \
  'http://localhost:8000/prediction' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "feature_vector": [
    -0.157953,-0.106749
  ],
  "score": true
}'
```

{
  "is_inlier": 1,
  "anomaly_score": -0.31314463447153723
}

```bash
curl -X 'GET' \
  'http://localhost:8000/model_information' \
  -H 'accept: application/json'
```

{
  "bootstrap": false,
  "contamination": 0.001,
  "max_features": 1,
  "max_samples": "auto",
  "n_estimators": 100,
  "n_jobs": null,
  "random_state": 16,
  "verbose": 0,
  "warm_start": false
}