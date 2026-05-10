#!/usr/bin/env bash
# Build API image with Podman and deploy to Kubernetes (local cluster).
# Usage:
#   ./scripts/k8s_deploy_podman.sh build              # only podman build
#   ./scripts/k8s_deploy_podman.sh load-minikube      # build + load into minikube + kubectl apply
#   ./scripts/k8s_deploy_podman.sh load-kind          # build + kind load + kubectl apply (KIND_CLUSTER=name)
#   ./scripts/k8s_deploy_podman.sh load-docker-desktop # build + docker load + kubectl apply (needs Docker CLI + Docker Desktop K8s)
#   ./scripts/k8s_deploy_podman.sh apply              # kubectl apply only (image must already exist in cluster)
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE_SHORT="heart-disease-api:local"
TAR="/tmp/heart-disease-api-image.tar"

build_image() {
  echo "==> Podman build ($IMAGE_SHORT) ..."
  podman build -f docker/Dockerfile -t "${IMAGE_SHORT}" .
  # Podman often also lists localhost/${IMAGE_SHORT}; ensure canonical tag for YAML
  podman tag "${IMAGE_SHORT}" "localhost/${IMAGE_SHORT}" 2>/dev/null || true
  echo "==> podman images:"
  podman images "${IMAGE_SHORT}" "localhost/${IMAGE_SHORT}" 2>/dev/null || podman images | head -5
}

case "${1:-help}" in
  build)
    build_image
    echo "Done. Load this image into your cluster runtime, then: kubectl apply -f k8s/"
    ;;
  load-minikube)
    command -v minikube >/dev/null || { echo "Install minikube first."; exit 1; }
    build_image
    echo "==> Saving image and loading into minikube..."
    podman save "${IMAGE_SHORT}" -o "${TAR}"
    minikube image load "${TAR}"
    rm -f "${TAR}"
    # Podman often records localhost/heart-disease-api:local; YAML uses heart-disease-api:local.
    if command -v docker >/dev/null && eval "$(minikube docker-env)" 2>/dev/null; then
      if docker image inspect localhost/heart-disease-api:local >/dev/null 2>&1; then
        docker tag localhost/heart-disease-api:local "${IMAGE_SHORT}"
      fi
      eval "$(minikube docker-env -u)"
    fi
    echo "==> kubectl apply..."
    kubectl apply -f "${ROOT}/k8s/deployment.yaml"
    kubectl apply -f "${ROOT}/k8s/service.yaml"
    echo "==> Status:"
    kubectl rollout status deployment/heart-disease-api --timeout=180s || true
    kubectl get pods,svc -l app=heart-disease-api
    echo ""
    echo "If EXTERNAL-IP stays pending, run in another terminal:"
    echo "  kubectl port-forward svc/heart-disease-api 8080:80"
    ;;
  load-kind)
    command -v kind >/dev/null || { echo "Install kind first."; exit 1; }
    CLUSTER="${KIND_CLUSTER:-kind}"
    build_image
    echo "==> Loading into kind cluster '${CLUSTER}'..."
    podman save "${IMAGE_SHORT}" -o "${TAR}"
    kind load image-archive "${TAR}" --name "${CLUSTER}"
    rm -f "${TAR}"
    kubectl apply -f "${ROOT}/k8s/deployment.yaml"
    kubectl apply -f "${ROOT}/k8s/service.yaml"
    kubectl rollout status deployment/heart-disease-api --timeout=180s || true
    kubectl get pods,svc -l app=heart-disease-api
    ;;
  load-docker-desktop)
    command -v docker >/dev/null || { echo "Need Docker CLI + Docker Desktop running to load image into Docker's store."; exit 1; }
    build_image
    echo "==> Export from Podman and docker load (for Docker Desktop Kubernetes)..."
    podman save "${IMAGE_SHORT}" -o "${TAR}"
    docker load -i "${TAR}"
    rm -f "${TAR}"
    kubectl apply -f "${ROOT}/k8s/deployment.yaml"
    kubectl apply -f "${ROOT}/k8s/service.yaml"
    kubectl rollout status deployment/heart-disease-api --timeout=180s || true
    kubectl get pods,svc -l app=heart-disease-api
    ;;
  apply)
    kubectl apply -f "${ROOT}/k8s/deployment.yaml"
    kubectl apply -f "${ROOT}/k8s/service.yaml"
    kubectl get pods,svc -l app=heart-disease-api
    ;;
  *)
    echo "Heart Disease API — Kubernetes deploy helper (Podman)"
    echo ""
    echo "Pick ONE path matching your cluster:"
    echo "  $0 build               Build image only"
    echo "  $0 load-minikube       Minikube + Podman"
    echo "  $0 load-kind           Kind + Podman (set KIND_CLUSTER if not 'kind')"
    echo "  $0 load-docker-desktop Docker Desktop K8s (Podman build → docker load)"
    echo "  $0 apply               Apply YAML only (image must exist in cluster)"
    echo ""
    echo "Then reach the API:"
    echo "  kubectl port-forward svc/heart-disease-api 8080:80"
    echo "  curl http://127.0.0.1:8080/health"
    exit 0
    ;;
esac
