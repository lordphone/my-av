#!/usr/bin/env python3
"""
Submit AV training job to Vertex AI
"""
from google.cloud import aiplatform

# Your project configuration
PROJECT_ID = "my-av-462300"
REGION = "us-central1"
CONTAINER_URI = "us-central1-docker.pkg.dev/my-av-462300/my-av/av-training:latest"

# Update these paths to match your actual GCS bucket and dataset location
BUCKET_NAME = "my-av"  # Replace with your actual bucket
DATASET_PATH = f"gs://{BUCKET_NAME}/datasets/comma2k19"  # Path to your dataset
CHECKPOINT_DIR = f"gs://{BUCKET_NAME}/checkpoints"
MODEL_DIR = f"gs://{BUCKET_NAME}/models"
LOG_DIR = f"gs://{BUCKET_NAME}/logs"

def submit_training_job():
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Create custom container training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name="av-training-job",
        container_uri=CONTAINER_URI,
        requirements=["nvidia-driver-version-470"],  # For GPU support
    )
    
    print(f"Submitting training job...")
    print(f"Container: {CONTAINER_URI}")
    print(f"Dataset: {DATASET_PATH}")
    
    # Submit the job
    job.run(
        args=[
            "--dataset-path", DATASET_PATH,
            "--checkpoint-dir", CHECKPOINT_DIR,
            "--model-dir", MODEL_DIR,
            "--log-dir", LOG_DIR,
            "--batch-size", "32",
            "--num-workers", "4",
            "--epochs", "10",  # Start with fewer epochs for testing
            "--mode", "train"
        ],
        replica_count=1,
        machine_type="n1-highmem-8",
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=1,
        boot_disk_size_gb=100,
        sync=False  # Run asynchronously
    )
    
    print(f"Training job submitted successfully!")
    print(f"Job name: {job.display_name}")
    print(f"You can monitor it in the GCP Console or with: gcloud ai custom-jobs list --region={REGION}")

if __name__ == "__main__":
    submit_training_job() 