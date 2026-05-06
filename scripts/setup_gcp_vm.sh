#!/usr/bin/env bash
# Bootstrap a fresh GCP VM (Ubuntu 22.04 / 24.04) for MultiWorld.
#
# Installs:
#   - NVIDIA driver (open kernel module, current production branch)
#   - Docker CE
#   - NVIDIA Container Toolkit  (so `--gpus all` works inside docker)
#
# Tested against:
#   - Ubuntu 22.04 LTS  + L4/A100/H100
#   - Ubuntu 24.04 LTS  + L4/A100/H100
#
# Usage (run as root, or with sudo):
#   bash scripts/setup_gcp_vm.sh
#
# Reboot is required after the driver is installed.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Re-running with sudo..."
  exec sudo -E bash "$0" "$@"
fi

UBUNTU_CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
echo "[setup] Detected Ubuntu codename: $UBUNTU_CODENAME"

# ---------------------------------------------------------------------------
# 1. NVIDIA driver
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
  echo "[setup] Installing NVIDIA driver..."
  apt-get update
  apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      gnupg \
      lsb-release \
      software-properties-common \
      ubuntu-drivers-common
  ubuntu-drivers install --gpgpu || ubuntu-drivers autoinstall

  # The --gpgpu metapackage installs the kernel module + libcompute but skips
  # the userspace tools, so `nvidia-smi` would be missing. Pull it in
  # explicitly. We discover the active driver branch from the kernel module
  # package that ubuntu-drivers just installed.
  DRV_BRANCH="$(dpkg -l 'linux-modules-nvidia-*-server-open-gcp' 2>/dev/null \
      | awk '/^ii/ {print $2}' | grep -oE 'nvidia-[0-9]+-server' | head -1 \
      | sed 's/nvidia-//;s/-server//')"
  if [[ -n "$DRV_BRANCH" ]]; then
    apt-get install -y --no-install-recommends "nvidia-utils-${DRV_BRANCH}-server" || true
  fi
  echo "[setup] NVIDIA driver installed. A REBOOT is required before docker --gpus works."
else
  echo "[setup] nvidia-smi already present, skipping driver install:"
  nvidia-smi || true
fi

# ---------------------------------------------------------------------------
# 2. Docker CE
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
  echo "[setup] Installing Docker CE..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $UBUNTU_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io \
      docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
else
  echo "[setup] Docker already installed: $(docker --version)"
fi

# ---------------------------------------------------------------------------
# 3. NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "[setup] Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
else
  echo "[setup] NVIDIA Container Toolkit already installed."
fi

# ---------------------------------------------------------------------------
# 4. Make sure the calling user can use docker without sudo
# ---------------------------------------------------------------------------
TARGET_USER="${SUDO_USER:-$USER}"
if [[ -n "$TARGET_USER" && "$TARGET_USER" != "root" ]]; then
  if ! id -nG "$TARGET_USER" | grep -qw docker; then
    usermod -aG docker "$TARGET_USER"
    echo "[setup] Added $TARGET_USER to the docker group. Log out / log in to apply."
  fi
fi

cat <<'EOF'

[setup] Done.

Next steps:
  1. If the NVIDIA driver was just installed, REBOOT the VM:
        sudo reboot
  2. After reboot, verify GPUs are visible to docker:
        docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  3. Download the model weights:
        bash scripts/download_models.sh
  4. Bring up the UI:
        bash scripts/run_ui.sh
EOF
