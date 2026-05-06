# Deploying the MultiWorld UI on a GCP VM

This guide walks through standing up the interactive Gradio tester
(`app/app.py`) on a fresh Google Cloud VM.

The tester wraps the same `WanVideoPipeline` used by `ittakestwo/parallel_inference.py`
but runs it on a single GPU, exposes the action space as form controls, and
streams the generated video back over HTTP.

---

## 0. What the UI does

* Pick a model checkpoint (`480p full`, `480p toy`, or `320p robots`).
* Upload an initial frame (a side-by-side stereo frame for It Takes Two,
  or a single robot view for the robotics model).
* For each player choose:
  * which discrete buttons to hold (`w/a/s/d/space/shift/ctrl/e/q/f`)
  * camera-look continuous values (`look_x`, `look_y` ∈ [-1, 1])
* Set the number of frames, inference steps, view mode, and seed.
* Click **Generate** — the model runs on the GPU and you get an mp4 back.

The UI is built so the first call lazy-loads the pipeline (a few minutes,
including ~25 GB of weights moving to VRAM); subsequent calls reuse the
in-memory model. Switching to a different checkpoint reloads it.

---

## 1. Hardware

The model is Wan2.2-TI2V-5B + a MultiWorld head. In bf16 the runtime
footprint is roughly **22 – 28 GB of VRAM** depending on resolution. Verified
working configurations:

| GPU                         | VRAM | Status                            |
|-----------------------------|------|-----------------------------------|
| NVIDIA L4                   | 24G  | Works for 320p robots; tight for 480p |
| NVIDIA A100 (40 / 80 GB)    | 40+G | Recommended                       |
| NVIDIA H100                 | 80G  | Fastest                           |
| NVIDIA T4                   | 16G  | **Not enough VRAM**               |

Disk: at least **300 GB** to comfortably hold the Wan2.2 base (~15 GB) and
the three MultiWorld checkpoints (~10 GB each).

---

## 2. Provision the VM

There is a helper that wraps `gcloud compute instances create`:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT

bash scripts/create_gcp_vm.sh \
  --project YOUR_PROJECT \
  --name multiworld-ui \
  --zone us-central1-a \
  --machine-type g2-standard-8 \
  --accelerator "type=nvidia-l4,count=1" \
  --disk-size 500
```

The script also opens TCP port 7860 in your firewall (tagged
`multiworld-ui`) so you can hit the Gradio UI from your laptop.

If you already have a VM, skip this step and SSH into it directly.

---

## 3. SSH in and bootstrap the system

```bash
gcloud compute ssh multiworld-ui --zone us-central1-a

# On the VM:
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/<you>/multi-world-cursor.git ~/multiworld
cd ~/multiworld

bash scripts/setup_gcp_vm.sh    # installs NVIDIA driver + Docker + nvidia-container-toolkit
sudo reboot                      # required after the driver install
```

After the reboot:

```bash
gcloud compute ssh multiworld-ui --zone us-central1-a
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU. If you don't, the rest of the guide will not work.

---

## 4. Download model weights

Get a Hugging Face token from <https://huggingface.co/settings/tokens> (or a
ModelScope token) and either export it to your shell or store it in
`~/.cache/huggingface/token` via `huggingface-cli login`:

```bash
cd ~/multiworld
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxx
bash scripts/download_models.sh                  # auto-detects HF
# or:
bash scripts/download_models.sh --source modelscope
```

This populates:

```
checkpoints/multiworld_480p_fulldata.safetensors
checkpoints/multiworld_480p_toydata.safetensors
checkpoints/multiworld_320p_robots.safetensors
models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors
```

Plan for ~30 – 40 GB of downloads. You can pass `--skip-base` or `--skip-ckpt`
on subsequent runs.

---

## 5. Run the UI

```bash
bash scripts/run_ui.sh
```

This builds the Docker image (the first build takes ~10 minutes — it pulls
the NVIDIA PyTorch image and installs the requirements) and starts the
container. The Gradio app is then reachable at:

```
http://<external-ip>:7860
```

To run detached:

```bash
bash scripts/run_ui.sh up -d
bash scripts/run_ui.sh logs -f
bash scripts/run_ui.sh down
```

If you'd rather not deal with firewall rules, edit `app/app.py`'s launch
arguments or pass `--share` to get a public `*.gradio.live` tunnel:

```bash
docker compose run --service-ports multiworld-ui \
    python -m app.app --share
```

---

## 6. Running without Docker (advanced)

Inside the VM (after `setup_gcp_vm.sh`):

```bash
conda create -n multiworld python=3.13 -y
conda activate multiworld
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r app/requirements.txt
python -m app.app
```

---

## 7. Troubleshooting

* **`Inference failed: ... CUDA out of memory`** — switch to the
  `Robots — 320p` model or shrink `Number of frames`. T4-class GPUs are
  not enough; you need at least an L4 (and ideally an A100/H100).
* **`A required checkpoint or config is missing on disk`** — the UI's model
  dropdown only includes entries whose `.safetensors` it can see in
  `checkpoints/`. Re-run `scripts/download_models.sh`.
* **The browser hangs at `Loading pipeline ...`** — the first call really
  does take a few minutes. Watch `bash scripts/run_ui.sh logs` for the
  `WanVideoPipeline loaded.` message. It's cached after that.
* **`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`** —
  the NVIDIA Container Toolkit isn't active. Re-run `setup_gcp_vm.sh` and
  reboot.

---

## 8. What's _not_ wired up

* Multi-GPU sequence-parallel inference (the original
  `parallel_inference.py` uses 8x sharding via `torchrun`). The UI is
  single-GPU only. For batch evaluations keep using
  `python -m torch.distributed.run --nproc_per_node=8 ittakestwo/parallel_inference.py ...`.
* Dataset-driven evaluation (the UI accepts a single image; for the full
  validation loop use the original script).
* Autoregressive long-rollout mode (`--inference-mode autoregressive`).
