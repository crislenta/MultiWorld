#!/usr/bin/env bash
# Provision a GCP VM suitable for running MultiWorld inference.
#
# Run this on YOUR LOCAL MACHINE (not on the VM). Requires `gcloud`
# already authenticated against the project where you want the instance.
#
# Defaults aim for a single L4 GPU which is the cheapest option that fits
# the 5B-parameter Wan2.2 + MultiWorld stack at bf16.
# For faster generation use --machine-type g2-standard-12 (1xL4),
# a2-highgpu-1g (1xA100 40G), or a3-highgpu-1g (1xH100 80G).
#
# Usage:
#   bash scripts/create_gcp_vm.sh \
#       --project my-gcp-project \
#       [--name multiworld-ui] \
#       [--zone us-central1-a] \
#       [--machine-type g2-standard-8] \
#       [--accelerator "type=nvidia-l4,count=1"] \
#       [--disk-size 500]
#
# After it boots:
#   gcloud compute ssh multiworld-ui --zone us-central1-a
#   git clone https://github.com/<you>/multi-world-cursor /workspace
#   cd /workspace
#   bash scripts/setup_gcp_vm.sh
#   sudo reboot
#   # ... reconnect ...
#   bash scripts/download_models.sh
#   bash scripts/run_ui.sh

set -euo pipefail

PROJECT=""
NAME="multiworld-ui"
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-8"
ACCELERATOR="type=nvidia-l4,count=1"
DISK_SIZE="500"
IMAGE_FAMILY="ubuntu-2404-lts-amd64"
IMAGE_PROJECT="ubuntu-os-cloud"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)        PROJECT="$2"; shift 2 ;;
    --name)           NAME="$2"; shift 2 ;;
    --zone)           ZONE="$2"; shift 2 ;;
    --machine-type)   MACHINE_TYPE="$2"; shift 2 ;;
    --accelerator)    ACCELERATOR="$2"; shift 2 ;;
    --disk-size)      DISK_SIZE="$2"; shift 2 ;;
    --image-family)   IMAGE_FAMILY="$2"; shift 2 ;;
    --image-project)  IMAGE_PROJECT="$2"; shift 2 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "--project is required" >&2
  exit 1
fi

echo "[gcp] Creating $NAME in $ZONE ($PROJECT) with $ACCELERATOR ..."

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_SIZE}GB" \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --metadata=enable-oslogin=TRUE \
  --tags=multiworld-ui

echo "[gcp] Opening port 7860 for the Gradio UI (firewall rule 'allow-multiworld-ui')..."
gcloud compute firewall-rules describe allow-multiworld-ui --project="$PROJECT" >/dev/null 2>&1 || \
  gcloud compute firewall-rules create allow-multiworld-ui \
    --project="$PROJECT" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:7860 \
    --target-tags=multiworld-ui

EXTERNAL_IP=$(gcloud compute instances describe "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

cat <<EOF

[gcp] Instance is up.
       External IP : $EXTERNAL_IP
       SSH         : gcloud compute ssh $NAME --project=$PROJECT --zone=$ZONE
       UI URL      : http://$EXTERNAL_IP:7860  (after you finish setup)

Next:
  gcloud compute ssh $NAME --project=$PROJECT --zone=$ZONE
  # then on the VM:
  sudo apt-get update && sudo apt-get install -y git
  git clone <your-fork-url> ~/multiworld && cd ~/multiworld
  bash scripts/setup_gcp_vm.sh
  sudo reboot
  # reconnect
  cd ~/multiworld
  HUGGING_FACE_HUB_TOKEN=hf_xxx bash scripts/download_models.sh
  bash scripts/run_ui.sh
EOF
