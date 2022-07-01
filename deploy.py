from google.cloud import aiplatform

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="opendataportal-imported",
    # artifact_uri="dps-opendataportal-354609-bucket/opendataportal/model",
    artifact_uri="gs://dps-opendataportal-354609-bucket/opendataportal/model",
    # artifact_uri="gs://dps-opendataportal-354609-bucket/opendataportal-model",
    
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
)

# Deploy the above model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4"
)