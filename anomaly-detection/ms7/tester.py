import requests
import random
import pandas as pd

ML_MODEL_ENDPOINT = "http://localhost:8000/prediction"
ML_MODEL_INFO_ENDPOINT = "http://localhost:8000/model_information"


test_df = pd.read_csv("jupyter/test.csv")
df_len = len(test_df)
inlier = 0
outlier = 0

for index, row in test_df.iterrows():

    print('call=' + str(index) + '/' + str(df_len))

    data = {}
    data['feature_vector'] = [row['mean'], row['sd']]
    data['score'] = False

    p = random.random()
    if p <= 0.25:
        data['score'] = True

    res = requests.post(url=ML_MODEL_ENDPOINT, json=data)
    result = res.json()

    if result['is_inlier'] == -1:
        outlier += 1
    else:
        inlier += 1

    print(result)


res = requests.get(url=ML_MODEL_INFO_ENDPOINT)
print(res.json())

print('outlier=' + str(outlier))
print('inlier=' + str(inlier))
