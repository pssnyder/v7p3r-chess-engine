#!/bin/bash
# Deploy V7P3R Analytics to GCP Cloud Run (local build)
# Builds Docker image locally and pushes to GCR

set -e

PROJECT_ID="v7p3r-lichess-bot"
REGION="us-central1"
JOB_NAME="v7p3r-weekly-analytics"
IMAGE_NAME="gcr.io/$PROJECT_ID/$JOB_NAME"
SERVICE_ACCOUNT="v7p3r-analytics@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_JOB="v7p3r-analytics-weekly"

echo "=========================================="
echo "V7P3R Analytics Deployment to GCP (Local Build)"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Job: $JOB_NAME"
echo "=========================================="

# Set project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Configure docker to use gcloud as credential helper
echo "Configuring Docker for GCR..."
gcloud auth configure-docker

# Build Docker image locally
echo "Building Docker image locally..."
docker build -t $IMAGE_NAME .

# Push to GCR
echo "Pushing image to GCR..."
docker push $IMAGE_NAME

# Deploy as Cloud Run Job
echo "Deploying Cloud Run job..."
if gcloud run jobs describe $JOB_NAME --region $REGION &> /dev/null; then
    echo "Updating existing job..."
    gcloud run jobs update $JOB_NAME \
        --image $IMAGE_NAME \
        --region $REGION
else
    echo "Creating new job..."
    gcloud run jobs create $JOB_NAME \
        --image $IMAGE_NAME \
        --region $REGION \
        --service-account $SERVICE_ACCOUNT \
        --set-env-vars "PROJECT_ID=$PROJECT_ID" \
        --memory 2Gi \
        --cpu 2 \
        --max-retries 2 \
        --task-timeout 3600
fi

# Create Cloud Scheduler job (runs every Sunday at midnight UTC)
echo "Setting up Cloud Scheduler..."
if gcloud scheduler jobs describe $SCHEDULER_JOB --location $REGION &> /dev/null; then
    echo "Scheduler already exists"
else
    gcloud scheduler jobs create http $SCHEDULER_JOB \
        --location $REGION \
        --schedule "0 0 * * 0" \
        --time-zone "UTC" \
        --uri "https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method POST \
        --oauth-service-account-email $SERVICE_ACCOUNT
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Analytics job will run every Sunday at midnight UTC"
echo ""
echo "Manual run: gcloud run jobs execute $JOB_NAME --region $REGION --project $PROJECT_ID"
echo "Check logs: gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME\" --limit 50 --project $PROJECT_ID"
echo ""
