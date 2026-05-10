# Heart Disease MLOps (Cleveland UCI)

Binary classification pipeline with **scikit-learn**, **MLflow**, **FastAPI**, **Docker/Podman**, **Kubernetes** manifests, **GitHub Actions** CI, and **Prometheus** metrics.

---

## Contents

- [Prerequisites](#prerequisites)
- [1. Clone and environment](#1-clone-and-environment)
- [2. Data](#2-data)
- [3. Train the model](#3-train-the-model)
- [4. MLflow UI (optional)](#4-mlflow-ui-optional)
- [5. Lint and tests](#5-lint-and-tests)
- [6. Run the API locally](#6-run-the-api-locally)
- [7. Container (Docker or Podman)](#7-container-docker-or-podman)
- [8. Kubernetes](#8-kubernetes)
- [9. Monitoring](#9-monitoring)
- [10. CI/CD](#10-cicd)
- [11. Report PDF](#11-report-pdf)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10 or newer (CI uses **3.11**) |
| **Git** | To clone the repository |
| **Container runtime** (optional) | **Docker** or **Podman** for images; **Colima** gives a Docker socket on macOS without Docker Desktop |
| **Kubernetes** (optional) | `kubectl` + local cluster (**Minikube**, **Kind**, or Docker Desktop Kubernetes) |

All commands below assume your shell is in the project root:

```bash
cd heart-disease-mlops
```

---

## 1. Clone and environment

```bash
git clone <YOUR_REPO_URL>
cd heart-disease-mlops

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

You should now have `heart_mlops` and `api` importable. Keep the virtualenv activated for every new terminal session (`source .venv/bin/activate`).

**Conda (alternative):**

```bash
conda env create -f environment.yml
conda activate heart-mlops
pip install -r requirements-dev.txt
pip install -e .
```

---

## 2. Data

The Cleveland processed file is expected at:

`data/raw/processed.cleveland.data`

If it is missing, download it:

```bash
python scripts/download_data.py
```

---

## 3. Train the model

This fits **logistic regression** and **random forest**, compares hold-out **ROC-AUC**, logs to **MLflow**, and writes:

- `models/heart_classifier.joblib` — model used by the API  
- `models/training_meta.json` — chosen model name and metrics  
- `mlruns/` — MLflow file store (local only; optional to commit)  
- `artifacts/` — ROC plots (and EDA figures if you ran EDA)

```bash
# Full training (5-fold CV; slower)
python -m heart_mlops.train

# Quick training (3-fold CV; same as CI)
python -m heart_mlops.train --fast
```

**EDA figures** (class balance, histograms, correlation heatmap):

```bash
python -m heart_mlops.eda
```

Outputs go under `artifacts/eda/`.

---

## 4. MLflow UI (optional)

After training, browse experiments:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open **http://127.0.0.1:5000** in a browser.

If `mlflow: command not found`, use the same venv and either reinstall deps (`pip install -r requirements.txt`) or:

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

---

## 5. Lint and tests

```bash
ruff check src tests
pytest tests/ -v
```

---

## 6. Run the API locally

Training must have produced `models/heart_classifier.joblib` (step 3).

```bash
cd heart-disease-mlops
source .venv/bin/activate

uvicorn api.main:app --host 0.0.0.0 --port 8080
```

**If you see `ModuleNotFoundError: No module named 'api'`:**

- Run `pip install -e .` from `heart-disease-mlops`, or  
- `PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8080`, or  
- `./scripts/run_api.sh` (sets `PYTHONPATH` for you; may need `chmod +x scripts/run_api.sh`).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness |
| `/predict` | POST | JSON body: all **13** Cleveland features → `prediction`, `probability_positive`, `confidence`, `risk_level` (`no risk` / `low risk` / `medium risk` / `high risk`) |
| `/metrics` | GET | Prometheus metrics |

**Example requests:**

```bash
curl -s http://127.0.0.1:8080/health

curl -s -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Optional environment variables: `MODEL_PATH`, `LOG_LEVEL`.

---

## 7. Container (Docker or Podman)

From `heart-disease-mlops/`:

**Docker**

```bash
docker build -f docker/Dockerfile -t heart-disease-api:local .
docker run --rm -p 8080:8080 heart-disease-api:local
```

**Podman**

```bash
podman build -f docker/Dockerfile -t heart-disease-api:local .
podman run --rm -p 8080:8080 localhost/heart-disease-api:local
```

The image runs a fast training step at **build** time, then starts **uvicorn** on port **8080**.

---

## 8. Kubernetes

You need a working cluster context (`kubectl cluster-info`) and an image the cluster can run.

**Recommended on macOS without Docker Desktop:** start **Colima**, then **Minikube**:

```bash
colima start
minikube start --driver=docker
kubectl config use-context minikube
```

**Deploy** (build with Podman, load into Minikube, apply manifests):

```bash
chmod +x scripts/k8s_deploy_podman.sh
./scripts/k8s_deploy_podman.sh load-minikube
```

Then reach the service:

```bash
kubectl port-forward svc/heart-disease-api 8080:80
curl -s http://127.0.0.1:8080/health
```

More options (`load-kind`, `load-docker-desktop`, `apply` only) are documented in **`k8s/README.md`**.

---

## 9. Monitoring

- **`GET /metrics`** — Prometheus text format (see `monitoring/prometheus.yml` for a sample scrape config).  
- **Logs** — structured lines on stdout for each `/predict` (optional header `X-Request-ID`).

---

## 10. CI/CD

Workflow: `.github/workflows/ci.yml`

On push/PR to `main` or `master`:

**Job `lint-test-train`**

1. Install Python **3.11** and dependencies (`requirements-dev.txt` + editable install)  
2. **Ruff**  
3. **Pytest**  
4. **`python -m heart_mlops.train --fast`**  
5. Upload **`models/heart_classifier.joblib`** as a workflow artefact  

**Job `docker-build-test`** (runs only if the job above succeeds)

6. **`docker build -f docker/Dockerfile`** — image trains inside the build and runs **uvicorn**  
7. Starts the container and **`curl`** checks **`GET /health`** — logs appear in the Actions run if anything fails  

That workflow log is suitable **container build/test proof** for coursework (screenshot: Actions → latest run → expand **docker-build-test**).

---

## 11. Report exports (PDF & Word)

Same source file for everything: **`reports/MLOps_Assignment_Report.md`**.

**PDF** (Pandoc → HTML, Chrome → PDF):

```bash
chmod +x scripts/pdf_report.sh
./scripts/pdf_report.sh
```

Produces `reports/MLOps_Assignment_Report.html` and `reports/MLOps_Assignment_Report.pdf` (requires **Pandoc** and **Google Chrome** for PDF).

**Word (`.docx`)** — Pandoc only:

```bash
chmod +x scripts/docx_report.sh
./scripts/docx_report.sh
```

Produces `reports/MLOps_Assignment_Report.docx`. If SVG diagrams do not appear inside Word, install **`librsvg`** (provides `rsvg-convert`), e.g. `brew install librsvg` on macOS, then rerun the script.

---

## Repository layout

```text
heart-disease-mlops/
├── .github/workflows/ci.yml    # CI pipeline
├── data/raw/                     # Cleveland CSV
├── docker/Dockerfile
├── k8s/                          # deployment.yaml, service.yaml
├── models/                       # training_meta.json; *.joblib after train (gitignored)
├── monitoring/prometheus.yml
├── reports/                      # Assignment report (Markdown, diagrams, pdf/docx scripts)
├── scripts/                      # run_api, k8s deploy, pdf_report, docx_report, download_data
├── src/
│   ├── heart_mlops/              # train, eda, inference, pipeline
│   └── api/main.py               # FastAPI app
├── tests/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Generated folders (typical `.gitignore`): `.venv/`, `mlruns/`, `artifacts/`, `*.egg-info/`, `models/*.joblib`.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `No module named 'api'` | `cd` into `heart-disease-mlops`, `source .venv/bin/activate`, `pip install -e .` |
| Port **8080** busy | Use another port: `uvicorn api.main:app --host 0.0.0.0 --port 8081` |
| Docker daemon not running | macOS: start **Docker Desktop** or **`colima start`** |
| Minikube **ImagePullBackOff** | Podman builds often tag `localhost/heart-disease-api:local`; `./scripts/k8s_deploy_podman.sh` retags inside Minikube. See `k8s/README.md`. |
| `mlflow` not found | Use `python -m mlflow ...` inside the project venv |

---

## Dataset citation

Janosi et al., Detrano et al.; **UCI Heart Disease** (Cleveland processed) — [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
