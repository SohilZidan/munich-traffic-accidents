# Munich Traffic Accidents Analysis

## Data Visualization
In the following figures data are visualised:
| ![monthly](imgs/monthly.png) |
|:--:|
| *Fig.1 - Monthly* |

|![monthly-insgesamt](imgs/monthly-insgesamt.png)|
|:--:|
| *Fig.2 - Monthly detailed (insgesamt)* |

It is clear that only  ***Verkehrsunfalle*** category has the sub-caegory ***Auspraegung*** which could be rid of in case of training a machine learning model.

## Building a predictive model
First data is preprocessed as follows:
1. read data as a dataframe
2. rows with missing values dropped
3. keeping only the features: `MONATSZAHL`, `AUSPRAEGUNG`, `JAHR`, `MONAT`, `WERT`.
4. transform categorical data `MONATSZAHL`, `AUSPRAEGUNG` into new columns and drop the original columns.
5. min-max normalization is applied to `JAHR`, `MONAT`, `WERT`.
6. data train-test split
7. train a shallow neural network with 3 hidden layers and one output layer. As the output is normalized, the output activation function is sigmoid. You can see the learning curve in *fig.3*:
![learning-curve](https://storage.googleapis.com/dps-opendataportal-354609-bucket/train.png)
## Deployment
for training I use GCP Vertex AI following the steps in [here](https://github.com/SohilZidan/dps-vertex-ai)\
deploy the mode using the command:
```bash
python3 deploy.py | tee deploy-output.txt
```

## Prediction API
<!-- the current version requires input features normalization and output prediction denormalization (read [predict.py](predict.py)) -->
Input is fed to the endpoint with the following format:

* store input in a file `INPUT-JSON`:
    ```json
    {
        "instances": [
            [2021,1,1,0,0,0,1,0]
        ]
    }
    ```
* fill `ENDPOINT_ID` and `PROJECT_ID`:
    ```bash
    ENDPOINT_ID=""
    PROJECT_ID=""
    INPUT_DATA_FILE="INPUT-JSON"
    ```
* make a Request:
    ```bash
    curl \
    -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
    -d "@${INPUT_DATA_FILE}"
    ``
**!NOTE!** you may have to authenticate your google account using:
```
gcloud auth application-default login
```
