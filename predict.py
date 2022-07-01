
import pandas as pd
from trainer.misc import Normalization
from google.cloud import aiplatform

# BUCKET = "gs://dps-opendataportal-354609-bucket"

class Normalization:

  fields_to_normalize = ["JAHR", "MONAT"]

  def __init__(self, train_stats):
    train_stats_tmp = train_stats.copy()
    target_stats = train_stats_tmp.loc[:, train_stats_tmp.columns.isin(["WERT"])]
    train_stats_tmp = train_stats_tmp.loc[:, train_stats_tmp.columns.isin(self.fields_to_normalize)]

    self.target_stats = target_stats.transpose()
    self.train_stats = train_stats_tmp.transpose()

  def normalize_features(self, x):
    # return (x - train_stats['mean']) / train_stats['std']
    normed_train_data = x.copy()
    if isinstance(x, list):
        normed_train_data[:2] = (normed_train_data[:2] - self.train_stats['min']) / (self.train_stats['max']- self.train_stats['min'])
    else:
        normed_train_data[self.fields_to_normalize] = (normed_train_data[self.fields_to_normalize] - self.train_stats['min']) / (self.train_stats['max']- self.train_stats['min'])
    return normed_train_data

  def normalize_target(self, x):
    # return (x - target_stats['mean']) / target_stats['std']
    return (x - self.target_stats['min']) / (self.target_stats['max']- self.target_stats['min'])

  def denormalize_target(self, x):
    # return x * target_stats['std']["WERT"] + target_stats['mean']["WERT"]
    return x * (self.target_stats['max']["WERT"] - self.target_stats['min']["WERT"]) + self.target_stats['min']["WERT"]


endpoint = aiplatform.Endpoint(
    endpoint_name="projects/284595570276/locations/us-central1/endpoints/678732925872635904"
)

# Read stats for normalization
train_stats_url = "https://storage.googleapis.com/dps-opendataportal-354609-bucket/train_stats.pkl"
train_stats = pd.read_pickle(train_stats_url)
norm = Normalization(train_stats)

sample_raw = ['Alkoholunfälle',
'insgesamt',
'2020',
'01']

sample = {
    "JAHR": int(sample_raw[2]),
    "MONAT": int(sample_raw[3]),
    "Alkoholunfälle": int(sample_raw[0] == "Alkoholunfälle"),
    "Fluchtunfälle": int(sample_raw[0] == "Fluchtunfälle"),
    "Verkehrsunfälle": int(sample_raw[0] == "Verkehrsunfälle"),
    "Verletzte und Getötete": int(sample_raw[1] == "Verletzte und Getötete"),
    "insgesamt": int(sample_raw[1] == "insgesamt"),
    "mit Personenschäden": int(sample_raw[1] == "mit Personenschäden")
}

# normalize input
normalized_sample = norm.normalize_features(sample)
# get prediction
response = endpoint.predict([normalized_sample])
# denormalize output to get the real value
y_pred = norm.denormalize_target(response.predictions[0][0])

print('API response: ', response)
print('Predicted Value: ', )
