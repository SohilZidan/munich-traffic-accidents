from google.cloud import aiplatform

BUCKET = "gs://dps-challenge-354007-bucket"

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="mta-deployed",
    # artifact_uri="dps-opendataportal-354609-bucket/opendataportal/model",
    artifact_uri=BUCKET+"/mta/end2endmodel",
    # artifact_uri="gs://dps-opendataportal-354609-bucket/opendataportal-model",
    
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-6:latest"
)

# Deploy the above model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4"
)
