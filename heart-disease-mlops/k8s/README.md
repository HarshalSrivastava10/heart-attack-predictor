# Kubernetes manifests

Files:

- `deployment.yaml` — Deployment using image **`heart-disease-api:local`**, port **8080**, probes on **`/health`**
- `service.yaml` — **LoadBalancer** Service, port **80** → targetPort **8080**

## Quick deploy (Podman on Mac)

From repo root:

```bash
chmod +x scripts/k8s_deploy_podman.sh
```

### A. Minikube

```bash
minikube start
kubectl config use-context minikube
./scripts/k8s_deploy_podman.sh load-minikube
kubectl port-forward svc/heart-disease-api 8080:80
```

### B. Kind

```bash
kind create cluster   # once
export KIND_CLUSTER=kind   # default cluster name
./scripts/k8s_deploy_podman.sh load-kind
kubectl port-forward svc/heart-disease-api 8080:80
```

### C. Docker Desktop Kubernetes

Requires **Docker CLI** + Docker Desktop with Kubernetes **enabled** (image must be in Docker’s image store):

```bash
kubectl config use-context docker-desktop
./scripts/k8s_deploy_podman.sh load-docker-desktop
kubectl port-forward svc/heart-disease-api 8080:80
```

### D. Only apply YAML (you loaded the image yourself)

```bash
./scripts/k8s_deploy_podman.sh apply
```

## Fix `kubectl` credentials errors

If `kubectl cluster-info` asks for **credentials**, your context points at a **cloud** cluster or an expired login:

```bash
kubectl config get-contexts
kubectl config use-context docker-desktop   # example for Docker Desktop
# or
kubectl config use-context minikube
```

## Verify

```bash
kubectl get pods -l app=heart-disease-api
curl -s http://127.0.0.1:8080/health    # after port-forward
```
