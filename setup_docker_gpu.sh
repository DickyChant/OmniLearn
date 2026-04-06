#!/usr/bin/env bash
# Setup Docker with NVIDIA GPU support on Ubuntu
# Usage: sudo bash setup_docker_gpu.sh
set -euo pipefail

echo "=== Step 1/4: Install Docker Engine ==="
apt-get update -qq
apt-get install -y -qq ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== Step 2/4: Install NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== Step 3/4: Add user to docker group ==="
REAL_USER="${SUDO_USER:-$USER}"
usermod -aG docker "$REAL_USER"

echo "=== Step 4/4: Verify GPU access ==="
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== Done! ==="
echo "Log out and back in (or run 'newgrp docker') for group changes to take effect."
echo ""
echo "To run the experiment:"
echo "  cd /home/sqian/Codes/OmniLearn"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace -w /workspace/scripts vmikuni/tensorflow:ngc-23.12-tf2-v1 bash run_peft_experiment.sh /workspace/experiment_root"
