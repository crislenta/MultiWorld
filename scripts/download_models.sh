#!/usr/bin/env bash
# Download MultiWorld checkpoints (and the Wan2.2 base model) into the
# expected on-disk layout.
#
# After running this you should have:
#   ./checkpoints/multiworld_480p_fulldata.safetensors
#   ./checkpoints/multiworld_480p_toydata.safetensors
#   ./checkpoints/multiworld_320p_robots.safetensors
#   ./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors
#
# Provide an authentication token via either:
#   HUGGING_FACE_HUB_TOKEN=...     (preferred, set via `huggingface-cli login` or env var)
#   MODELSCOPE_API_TOKEN=...       (fallback)
#
# Usage:
#   bash scripts/download_models.sh                # auto-pick HF if available, else ModelScope
#   bash scripts/download_models.sh --source hf
#   bash scripts/download_models.sh --source modelscope
#   bash scripts/download_models.sh --skip-base   # skip the Wan2.2 5B download

set -euo pipefail

SOURCE="auto"
SKIP_BASE=0
SKIP_CKPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)      SOURCE="$2"; shift 2 ;;
    --skip-base)   SKIP_BASE=1; shift ;;
    --skip-ckpt)   SKIP_CKPT=1; shift ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p checkpoints models

# Auto-detect source if not set explicitly.
if [[ "$SOURCE" == "auto" ]]; then
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] || [[ -f "$HOME/.cache/huggingface/token" ]]; then
    SOURCE="hf"
  elif [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
    SOURCE="modelscope"
  else
    SOURCE="hf"   # default; will prompt for login if no token present
  fi
fi
echo "[download] Source: $SOURCE"

# ---------------------------------------------------------------------------
# Make sure the chosen client is installed.
# ---------------------------------------------------------------------------
if [[ "$SOURCE" == "hf" ]]; then
  python3 -m pip install --quiet --upgrade "huggingface_hub[cli]"
elif [[ "$SOURCE" == "modelscope" ]]; then
  python3 -m pip install --quiet --upgrade modelscope
else
  echo "Unknown --source: $SOURCE" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. MultiWorld checkpoints
# ---------------------------------------------------------------------------
if [[ $SKIP_CKPT -eq 0 ]]; then
  echo "[download] Fetching MultiWorld checkpoints..."

  if [[ "$SOURCE" == "hf" ]]; then
    if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
      huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential || true
    fi
    for f in \
        multiworld_480p_fulldata.safetensors \
        multiworld_480p_toydata.safetensors \
        multiworld_320p_robots.safetensors ; do
      if [[ -f "checkpoints/$f" ]]; then
        echo "[download]   - $f already present, skipping."
        continue
      fi
      echo "[download]   - $f"
      huggingface-cli download Haoyuwu/MultiWorldCheckpoint "$f" \
        --local-dir checkpoints --repo-type model
    done
  else
    if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
      modelscope login --token "$MODELSCOPE_API_TOKEN" || true
    fi
    for f in \
        multiworld_480p_fulldata.safetensors \
        multiworld_480p_toydata.safetensors \
        multiworld_320p_robots.safetensors ; do
      if [[ -f "checkpoints/$f" ]]; then
        echo "[download]   - $f already present, skipping."
        continue
      fi
      echo "[download]   - $f"
      modelscope download --model HaoyuWuRUC/MultiWorldCheckpoint \
        "$f" --local_dir checkpoints
    done
  fi
fi

# ---------------------------------------------------------------------------
# 2. Wan2.2 TI2V-5B base model
# ---------------------------------------------------------------------------
if [[ $SKIP_BASE -eq 0 ]]; then
  echo "[download] Fetching Wan-AI/Wan2.2-TI2V-5B base weights..."
  TARGET="models/Wan-AI/Wan2.2-TI2V-5B"
  mkdir -p "$TARGET"

  if [[ "$SOURCE" == "hf" ]]; then
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
      --include "diffusion_pytorch_model*.safetensors" \
      --include "*.json" \
      --local-dir "$TARGET" --repo-type model
  else
    modelscope download --model Wan-AI/Wan2.2-TI2V-5B \
      "diffusion_pytorch_model*.safetensors" "*.json" \
      --local_dir "$TARGET"
  fi
fi

echo "[download] Done."
ls -lh checkpoints || true
echo "---"
ls -lh "models/Wan-AI/Wan2.2-TI2V-5B" 2>/dev/null || true
