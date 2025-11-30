#!/bin/bash
# Deploy V7P3R Analytics to GCP
# This script sets up the analytics system as a Cloud Run job with Cloud Scheduler

set -e

PROJECT_ID="v7p3r-lichess-bot"
REGION="us-central1"
JOB_NAME="v7p3r-weekly-analytics"
SERVICE_ACCOUNT="v7p3r-analytics@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_JOB="v7p3r-analytics-weekly"

echo "=========================================="
echo "V7P3R Analytics Deployment to GCP"
echo "=========================================="

# Check if gcloud is configured
if ! gcloud config get-value project &> /dev/null; then
    echo "Error: gcloud not configured. Run 'gcloud auth login' first."
    exit 1
fi

# Set project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create service account if it doesn't exist
echo "Creating service account..."
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT &> /dev/null; then
    gcloud iam service-accounts create v7p3r-analytics \
        --display-name="V7P3R Analytics Service Account"
    
    # Grant necessary permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/compute.instanceAdmin"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/storage.objectAdmin"
    
    echo "Service account created and configured"
else
    echo "Service account already exists"
fi

# Build Docker image
echo "Building Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$JOB_NAME .

# Deploy as Cloud Run Job
echo "Deploying Cloud Run job..."
gcloud run jobs create $JOB_NAME \
    --image gcr.io/$PROJECT_ID/$JOB_NAME \
    --region $REGION \
    --service-account $SERVICE_ACCOUNT \
    --set-env-vars "PROJECT_ID=$PROJECT_ID" \
    --memory 2Gi \
    --cpu 2 \
    --max-retries 2 \
    --task-timeout 3600 \
    || gcloud run jobs update $JOB_NAME \
        --image gcr.io/$PROJECT_ID/$JOB_NAME \
        --region $REGION

# Create Cloud Scheduler job (runs every Monday at 9 AM)
echo "Setting up Cloud Scheduler..."
gcloud scheduler jobs create http $SCHEDULER_JOB \
    --location $REGION \
    --schedule "0 9 * * 1" \
    --time-zone "America/New_York" \
    --uri "https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
    --http-method POST \
    --oauth-service-account-email $SERVICE_ACCOUNT \
    || gcloud scheduler jobs update http $SCHEDULER_JOB \
        --location $REGION \
        --schedule "0 9 * * 1"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Analytics job will run every Monday at 9 AM EST"
echo ""
echo "Manual run: gcloud run jobs execute $JOB_NAME --region $REGION"
echo "Check logs: gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME\" --limit 50"
echo ""
echo "Next steps:"
echo "1. Configure email settings in Cloud Run job environment"
echo "2. Test with: gcloud run jobs execute $JOB_NAME --region $REGION"
echo "3. Monitor first scheduled run on Monday"
echo ""
