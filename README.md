# AV Model Training

This project provides utilities for training an autonomous vehicle (AV) model. The recommended approach is to use the provided Docker environment so you do not need to install the deep learning stack locally.

## Prerequisites

- **Docker** with GPU support (e.g. NVIDIA Container Toolkit)
- A copy of the `comma2k19` dataset or your own dataset organised in the same structure

## Building the Docker image

Clone this repository and build the image:

```bash
# from the repository root
docker build -t av-trainer .
```

The build step creates a Conda environment defined in `environment.yml` and copies the project sources into `/app` inside the image. The container automatically runs all commands inside this environment.

## Running the training container

1. Place your dataset on the host machine. In the examples below we assume it lives in `~/data/comma2k19`.
2. Start the container and mount the dataset directory. The dataset path inside the container is passed through the `DATASET_PATH` environment variable. The startup script checks that this directory exists before training starts.

```bash
docker run --gpus all \
  -v ~/data/comma2k19:/app/data \
  -e DATASET_PATH=/app/data \
  -e NUM_WORKERS=4 \
  -e BATCH_SIZE=32 \
  av-trainer
```

The `run_training.sh` script invoked by the container reads these variables and forwards them to `src/training/train.py`. You can supply additional arguments by setting environment variables or editing the command line in the script.

### Customising parameters

- `DATASET_PATH` – location of the dataset inside the container. Must match the mounted volume.
- `NUM_WORKERS` – number of data loading workers.
- `BATCH_SIZE` – batch size for training.
- `CHECKPOINT_DIR`, `MODEL_DIR`, `LOG_DIR` – output locations for training artefacts.

If you want to run training manually you can execute the script directly:

```bash
bash scripts/run_training.sh --dataset-path /path/to/data --num-workers 4 --batch-size 32
```

All parameters accepted by `train.py` can be passed on the command line.

## Repository structure

- `src/` – source code for data loading, models and training.
- `scripts/` – helper scripts including `run_training.sh` used inside Docker.
- `tests/` – unit tests.
- `environment.yml` – Conda environment specification used in the Docker image.

## Cloud Deployment Guide

This section covers deploying the training pipeline to Google Cloud Platform (GCP) for scalable training.

### Prerequisites for Cloud Deployment

1. **GCP Account**: Active Google Cloud project with billing enabled
2. **Required APIs**: Enable these APIs in your GCP project:
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   gcloud services enable storage-api.googleapis.com
   ```
3. **Local Tools**:
   - `gcloud` CLI installed and authenticated
   - Docker with GCP authentication configured
4. **Storage**: Create GCS bucket for datasets and outputs

### Step 1: Build and Push Docker Image

#### 1.1 Set up Artifact Registry

Create a Docker repository in Artifact Registry:

```bash
# Set your project variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"  # Choose your preferred region
export REPO_NAME="av-trainers"

# Create repository
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="AV model training images"
```

#### 1.2 Configure Docker Authentication

```bash
# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker $REGION-docker.pkg.dev
```

#### 1.3 Build and Push Image

```bash
# Build image with cloud-compatible tag
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/av-training:latest .

# Push to Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/av-training:latest
```

### Step 2: Prepare Data and Storage

#### 2.1 Upload Dataset to Cloud Storage

```bash
# Create GCS bucket for your data
export BUCKET_NAME="your-av-training-bucket"
gsutil mb gs://$BUCKET_NAME

# Upload your dataset
gsutil -m cp -r /path/to/local/comma2k19 gs://$BUCKET_NAME/datasets/

# Create directories for outputs
gsutil mkdir gs://$BUCKET_NAME/models
gsutil mkdir gs://$BUCKET_NAME/checkpoints
gsutil mkdir gs://$BUCKET_NAME/logs
```

### Step 3: Cloud Training Options

#### Option 3A: Vertex AI Custom Training (Recommended)

Create a training job using Vertex AI:

```python
# training_job.py
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.CustomContainerTrainingJob(
    display_name="av-training",
    container_uri=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/av-training:latest",
    requirements=["nvidia-driver-version-470"],  # For GPU support
)

# Run training job
job.run(
    args=[
        "--dataset-path", f"gs://{BUCKET_NAME}/datasets/comma2k19",
        "--checkpoint-dir", f"gs://{BUCKET_NAME}/checkpoints",
        "--model-dir", f"gs://{BUCKET_NAME}/models", 
        "--log-dir", f"gs://{BUCKET_NAME}/logs",
        "--batch-size", "32",
        "--num-workers", "4",
        "--epochs", "50"
    ],
    replica_count=1,
    machine_type="n1-highmem-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=1,
    boot_disk_size_gb=100,
    sync=False  # Run asynchronously
)
```

#### Option 3B: Compute Engine with Managed Instance Groups

For more control over the training environment:

```bash
# Create instance template with GPU
gcloud compute instance-templates create-with-container av-training-template \
    --container-image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/av-training:latest \
    --container-arg="--dataset-path=gs://$BUCKET_NAME/datasets/comma2k19" \
    --container-arg="--checkpoint-dir=gs://$BUCKET_NAME/checkpoints" \
    --container-arg="--model-dir=gs://$BUCKET_NAME/models" \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --boot-disk-size=100GB \
    --scopes=cloud-platform

# Create managed instance group (for auto-scaling)
gcloud compute instance-groups managed create av-training-group \
    --template=av-training-template \
    --size=1 \
    --zone=us-central1-a
```

### Step 4: Monitoring and Logging

#### 4.1 View Training Logs

```bash
# Vertex AI logs
gcloud ai custom-jobs list
gcloud ai custom-jobs describe JOB_ID --region=$REGION

# Container logs
gcloud logging read "resource.type=vertex_ai_training"
```

#### 4.2 Monitor Resources

- **Cloud Console**: Monitor GPU utilization and costs
- **Cloud Monitoring**: Set up alerts for job failures
- **Vertex AI Dashboard**: Track training metrics

### Step 5: Retrieving Results

#### 5.1 Download Trained Models

```bash
# Download best model
gsutil cp gs://$BUCKET_NAME/models/best_model.pth ./

# Download checkpoints
gsutil -m cp -r gs://$BUCKET_NAME/checkpoints ./

# Download logs for analysis
gsutil -m cp -r gs://$BUCKET_NAME/logs ./
```

### Cost Optimization Tips

1. **Use Preemptible Instances**: Add `--preemptible` to reduce costs by ~70%
2. **Right-size Resources**: Start with smaller instances, scale up as needed
3. **Spot Instances**: For non-critical training runs
4. **Storage Classes**: Use cheaper storage classes for archival data
5. **Lifecycle Policies**: Auto-delete old checkpoints and logs

```bash
# Example: Create preemptible training job
job.run(
    # ... other parameters ...
    spot=True,  # Use spot instances
    max_run_duration=86400,  # 24 hour timeout
)
```

### Multi-GPU and Distributed Training

For large-scale training across multiple GPUs:

```python
# Distributed training configuration
job = aiplatform.CustomContainerTrainingJob(
    display_name="av-distributed-training",
    container_uri=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/av-training:latest",
)

job.run(
    # ... other args ...
    replica_count=2,  # Number of worker replicas
    machine_type="n1-highmem-16",
    accelerator_type="NVIDIA_TESLA_V100", 
    accelerator_count=4,  # 4 GPUs per replica
)
```

### Troubleshooting Common Issues

#### Build/Push Issues
- **Authentication**: Ensure `gcloud auth login` and Docker auth are configured
- **Permissions**: Check IAM roles (Artifact Registry Writer, Storage Admin)
- **Network**: Verify VPC and firewall rules allow Docker registry access

#### Training Issues
- **OOM Errors**: Reduce batch size or use gradient accumulation
- **Storage Errors**: Check GCS bucket permissions and path formatting
- **GPU Issues**: Verify CUDA drivers and GPU quotas in your region

#### Performance Issues
- **Slow Data Loading**: Increase `num_workers` and use regional storage
- **Network Bottlenecks**: Use Persistent Disks for frequently accessed data
- **Memory Issues**: Monitor memory usage and adjust machine types

### Security Best Practices

1. **IAM**: Use principle of least privilege for service accounts
2. **VPC**: Run training in private networks when possible
3. **Encryption**: Enable encryption at rest for storage and models
4. **Secrets**: Use Secret Manager for sensitive configuration

```bash
# Create service account for training
gcloud iam service-accounts create av-training-sa

# Grant minimal required permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:av-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### Example: Complete Training Pipeline

Here's a complete script to run distributed training:

```bash
#!/bin/bash
# deploy_training.sh

export PROJECT_ID="your-project-id"
export REGION="us-central1"
export REPO_NAME="av-trainers"
export BUCKET_NAME="your-av-training-bucket"

# Build and push
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/av-training:latest .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/av-training:latest

# Submit training job
python3 - << EOF
from google.cloud import aiplatform
aiplatform.init(project="$PROJECT_ID", location="$REGION")

job = aiplatform.CustomContainerTrainingJob(
    display_name="av-production-training",
    container_uri="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/av-training:latest"
)

job.run(
    args=[
        "--dataset-path", "gs://$BUCKET_NAME/datasets/comma2k19",
        "--checkpoint-dir", "gs://$BUCKET_NAME/checkpoints",
        "--model-dir", "gs://$BUCKET_NAME/models",
        "--log-dir", "gs://$BUCKET_NAME/logs",
        "--batch-size", "64",
        "--num-workers", "8",
        "--epochs", "100",
        "--mode", "train"
    ],
    replica_count=1,
    machine_type="n1-highmem-16",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=2,
    spot=True,
    max_run_duration=7200,  # 2 hours
)
EOF

echo "Training job submitted successfully!"
```

