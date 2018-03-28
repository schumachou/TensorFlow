from oauth2client.client import GoogleCredentials
import googleapiclient.discovery

# project info
PROJECT_ID = "tensorflow-class-199302"
MODEL_NAME = "earnings"
CREDENTIALS_FILE = "credentials.json"

# values for prediction
inputs_for_prediction = [
    {"input": [0.4999, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5]}
]

# Connect to the Google Cloud-ML Service
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)

# Connect to Prediction Model
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
response = service.projects().predict(
    name=name,
    body={'instances': inputs_for_prediction}
).execute()

# Report any errors
if 'error' in response:
    raise RuntimeError(response['error'])

# Grab the results from the response object
results = response['predictions']

# Print the results!
print(results)
