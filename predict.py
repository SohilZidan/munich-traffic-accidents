from google.cloud import aiplatform
from transform_input import transform
endpoint = aiplatform.Endpoint(
    endpoint_name="projects/166843564221/locations/us-central1/endpoints/7181396425144532992"
)

sample_raw = ['Alkoholunfälle',
'insgesamt',
'2021',
'01']

# sample = {
#     "JAHR": int(sample_raw[2]),
#     "MONAT": int(sample_raw[3]),
#     "Alkoholunfälle": int(sample_raw[0] == "Alkoholunfälle"),
#     "Fluchtunfälle": int(sample_raw[0] == "Fluchtunfälle"),
#     "Verkehrsunfälle": int(sample_raw[0] == "Verkehrsunfälle"),
#     "Verletzte und Getötete": int(sample_raw[1] == "Verletzte und Getötete"),
#     "insgesamt": int(sample_raw[1] == "insgesamt"),
#     "mit Personenschäden": int(sample_raw[1] == "mit Personenschäden")
# }


# convert to a valid input
sample = transform(sample_raw)

# get prediction
response = endpoint.predict([sample])
y_pred = response.predictions[0][0]

print('API response: ', response)
print('Predicted Value: ', y_pred)
