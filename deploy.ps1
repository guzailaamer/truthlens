# deploy.ps1 - TruthLens Cloud Run deployment for Windows PowerShell
# Usage: .\deploy.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Setting project: $ProjectId"
gcloud config set project $ProjectId

Write-Host "==> Enabling required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com logging.googleapis.com

Write-Host "==> Getting project number..."
$ProjectNumber = gcloud projects describe $ProjectId --format='value(projectNumber)'
$SA = "$ProjectNumber-compute@developer.gserviceaccount.com"

Write-Host "==> Granting Vertex AI role to: $SA"
gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$SA" --role="roles/aiplatform.user" --condition=None

Write-Host "==> Building image via Cloud Build..."
gcloud builds submit --tag "gcr.io/$ProjectId/truthlens"

Write-Host "==> Deploying to Cloud Run..."
$EnvVars = "GCP_PROJECT=$ProjectId,GCP_REGION=global"
gcloud run deploy truthlens --image "gcr.io/$ProjectId/truthlens" --platform managed --region $Region --allow-unauthenticated --set-env-vars $EnvVars --max-instances 3 --memory 512Mi --concurrency 80 --timeout 60

Write-Host "==> Done. Service URL:"
gcloud run services describe truthlens --region $Region --format='value(status.url)'
