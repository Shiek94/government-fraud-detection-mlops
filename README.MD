# Government Fraud Detection — MLOps

An educational MLOps prototype for screening financial-assistance applications at a fictional government agency. The system estimates fraud probability and flags selected applications for human review. It does **not** establish fraud or automatically reject applications.

The project uses entirely **synthetic application data**. The agency portal, caseworker workflow, and real applicant-data ingestion are conceptual integration points, not implemented applications.

## Architecture

- **Online inference:** An IAM-authenticated AWS Lambda Function URL invokes a FastAPI application packaged in a Lambda container. The service loads an approved scikit-learn model artifact from Amazon S3 and returns a fraud probability.
- **Application image delivery:** A GitHub Actions workflow builds and pushes the API container image to Amazon ECR. Deployment of that image to Lambda is a separate operation.
- **Model maintenance:** A different GitHub Actions workflow simulates monthly retraining, compares candidate performance, persists state in S3, refreshes Lambda's model configuration, and verifies the live API.
- **Monitoring:** MLflow records experiments and model/drift metrics; Amazon CloudWatch records Lambda's operational logs and metrics.

See [System Architecture](docs/architecture.md) for the online and offline diagrams and implementation boundaries.

## Technology stack

Python 3.10; pandas; NumPy; scikit-learn; joblib; FastAPI and Pydantic; Mangum; MLflow; pytest; Docker; GitHub Actions; AWS Lambda, Lambda Function URLs, S3, ECR, IAM, and CloudWatch.

## Synthetic data and model

The generator produces a baseline dataset for **December 2025** and new datasets for **January through December 2026**. The initial dataset contains 2,000 applications; each simulated month adds 1,000. The replay adds the 800 monthly training observations to its cumulative training data and retains a 200-observation monthly evaluation set. The expected final cumulative training size is **11,200 rows**, subject to verification after all 12 runs.

The model uses these input features:

| Feature | Type |
| --- | --- |
| monthly_income | Nonnegative integer |
| requested_amount | Positive integer |
| household_size | Positive integer |
| previous_applications | Nonnegative integer |
| employment_status | employed, unemployed, or self-employed |

The synthetic target is fraud_confirmed. A scikit-learn Pipeline one-hot-encodes employment_status and fits a DecisionTreeClassifier with max_depth=3. The baseline classification threshold is **0.25**. The API returns the estimated class-1 probability and flags applications at or above the threshold.

During replay, the candidate and currently approved models are scored on the same monthly evaluation set. **A candidate is promoted only when its F1 score is strictly higher.** Otherwise, the approved model remains unchanged.

## Local reproduction (Windows PowerShell)

An AWS account is **not** required to train the baseline model and exercise the local API.

Prerequisites: Python 3.10 and Git.

```powershell
git clone https://github.com/Shiek94/government-fraud-detection-mlops.git
cd government-fraud-detection-mlops
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"

# For local training, leave REPLAY_WORKSPACE and MODEL_S3_BUCKET unset.
python -m fraud_detection.train_model
python -m pytest -v
python -m uvicorn fraud_detection.api:app --app-dir src --reload
```

The final command starts the development server on http://127.0.0.1:8000. Keep that terminal open; use a second PowerShell terminal for API requests.

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"

$requestBody = @{
    monthly_income = 1800
    requested_amount = 5000
    household_size = 3
    previous_applications = 2
    employment_status = "unemployed"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $requestBody
```

The exact probability depends on the saved model. The prediction response contains fraud_probability, flagged_for_review, and model_version.

### REST endpoints

| Endpoint | Purpose |
| --- | --- |
| GET /health | Basic availability check; returns status OK |
| POST /validate | Validates an application payload |
| POST /predict | Returns fraud probability, review flag, and model version |

FastAPI/Pydantic validates the request schema and rejects invalid values. Authentication of the **deployed** API is enforced by the AWS Lambda Function URL using AWS_IAM and signed requests (SigV4). A local Uvicorn development server does not provide that AWS authentication boundary.

### Local MLflow UI

Run this in a separate terminal from the repository root:

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000. This reads the local baseline tracking database. The automated replay uses a separate database in its isolated workspace; its SQLite database and mlruns artifacts are persisted to S3 between GitHub Actions runs and must be restored before browsing that history locally.

## Automated 12-month replay

Workflow: .github/workflows/replay-12-months.yml

- The workflow can run from a GitHub Actions schedule or be started with workflow_dispatch.
- The accelerated demonstration schedule is currently `4-59/5 * * * *` (one simulated month per triggered execution, not one real calendar month).
- Scheduled replay jobs execute only when the repository variable FRAUD_REPLAY_ENABLED is set to the string true. Manual dispatch is permitted by the workflow's job condition.
- A concurrency group prevents simultaneous replay jobs.
- Each run restores its previous checkpoint, approved model, and MLflow state from S3. If no replay state exists, it initializes the December 2025 baseline before advancing to January 2026.
- It generates the next month's synthetic applications, measures drift, trains and evaluates a candidate, records MLflow metrics, and writes the new checkpoint.
- It persists the approved model to S3, refreshes Lambda environment variables MODEL_S3_KEY and MODEL_REVISION, and verifies authenticated GET /health and POST /predict requests.
- Once checkpoint_2026-12.joblib exists, later runs skip further monthly training. The **cron schedule itself continues** until the workflow is disabled or its repository-variable gate is turned off.

Replay state is stored under these S3 prefixes in the project bucket:

| Prefix | Contents |
| --- | --- |
| checkpoints/replay-2026/ | December 2025 baseline and monthly checkpoint files |
| models/replay-2026/ | Current approved replay model artifact |
| mlflow/replay-2026/ | MLflow SQLite database and run artifacts |

The initial model and non-replay deployment artifacts are kept separate from replay state.

### Drift monitoring

The replay compares each month's mean requested_amount with a reference mean calculated from the initial training data:

```text
relative_change = abs(monthly_mean - reference_mean) / reference_mean
drift_detected = relative_change >= 0.25
```

Beginning in simulated July 2026, requested amounts are multiplied by 1.5 to introduce a distribution shift. The workflow logs the measured drift to MLflow and its job output.

**Drift detection is monitoring evidence; it does not independently trigger retraining.** The scheduled replay already trains a candidate each simulated month.

## Monitoring and security

- **Model monitoring:** Monthly candidate/current F1 comparison and requested-amount drift metrics in MLflow.
- **Service monitoring:** AWS Lambda metrics and logs in Amazon CloudWatch, including invocation errors and execution duration.
- **API access:** AWS_IAM-protected Lambda Function URL; callers must have suitable AWS permissions and sign requests.
- **CI/CD access:** GitHub Actions uses AWS OIDC role assumption rather than long-lived AWS keys stored in the repository.
- **Data:** All applications are fictional. Model artifacts and replay state use restricted S3 access.

## Limitations

- Synthetic labels and features are illustrative and do not demonstrate real-world predictive validity, bias control, or fairness.
- Model quality is limited; fraud probabilities should support **human review**, never automatic benefit denial.
- The agency backend and caseworker queue are conceptual and have not been implemented.
- The drift monitor checks one feature's mean; it is not a comprehensive data-quality or statistical-drift monitoring service.
- Candidate rejection protects the approved model from a worse F1 score, but fully automatic rollback after a partial deployment failure is not implemented.
- Cloud infrastructure was provisioned for a temporary demonstration, not by a one-command infrastructure-as-code setup. Reproducing the cloud deployment requires a separately configured AWS environment and permissions.

## Project completion and cleanup

After confirming the December 2026 checkpoint, preserve evidence from GitHub Actions, MLflow, S3, CloudWatch, and the authenticated REST API. Then set FRAUD_REPLAY_ENABLED to false, disable the scheduled replay if no longer needed, and remove unnecessary AWS resources to avoid ongoing costs.
