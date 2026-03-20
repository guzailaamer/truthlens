#!/bin/bash
# deploy.sh — TruthLens Cloud Run deployment
# Usage: ./deploy.sh <GCP_PROJECT_ID> [GCP_REGION]
# Example: ./deploy.sh my-project-123 us-central1
set -eo pipefail

PROJECT_ID="$1"
REGION="${2:-us-central1}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: GCP_PROJECT_ID is required as the first argument." >&2
  echo "Usage: ./deploy.sh <GCP_PROJECT_ID> [GCP_REGION]" >&2
  exit 1
fi

echo "==> Setting project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "==> Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  logging.googleapis.com

echo "==> Granting Vertex AI access to default compute service account..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# IAM binding failure is printed as a warning — it may already exist or require elevated perms.
# The deployment will still proceed; re-run with Owner permissions if Vertex AI calls fail.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  2>&1 || echo "WARNING: IAM binding failed — check that you have resourcemanager.projects.setIamPolicy permission."

echo "==> Building and submitting image via Cloud Build..."
gcloud builds submit --tag "gcr.io/$PROJECT_ID/truthlens"

echo "==> Deploying to Cloud Run (region: $REGION)..."
gcloud run deploy truthlens \
  --image "gcr.io/$PROJECT_ID/truthlens" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT=$PROJECT_ID,GCP_REGION=global" \
  --max-instances 3 \
  --memory 512Mi \
  --concurrency 80 \
  --timeout 60

echo ""
echo "==> Deployment complete. Service URL:"
gcloud run services describe truthlens --region "$REGION" --format='value(status.url)'
