# OpenVLA — Docker Install Guide (RTX 5090 / Blackwell)

A complete, battle-tested installation guide for **OpenVLA 7B** on Docker — including real fixes for RTX 5090 Blackwell (sm_120) and corporate firewall SSL issues not covered in the official docs.

📖 **Full interactive guide → [View on GitHub Pages](https://your-username.github.io/openvla-install-guide/)**

---

## Tested On

| | |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 5090 (Blackwell / sm_120) |
| CUDA | 12.8 |
| Python | 3.10 |
| PyTorch | 2.12 nightly (cu128) |
| Docker | NVIDIA Container Toolkit |

---

## What's Fixed Here

Issues not covered in the official OpenVLA docs:

- 🔴 **RTX 5090 (sm_120) not supported by stable PyTorch** — stable PyTorch crashes at runtime with `no kernel image` error. Fix: use PyTorch nightly `cu128` with `TORCH_CUDA_ARCH_LIST="12.0"`
- 🔴 **SSL certificate errors behind corporate firewall** — pip and git fail with `CERTIFICATE_VERIFY_FAILED`. Fix: install company cert into the Docker image before any other command
- 🔴 **Old PyTorch (cu121) cached in Docker layer** — rebuilding without `--no-cache` silently reuses the wrong version. Fix: delete old image with `docker rmi` first
- 🔴 **Model weights (~14 GB) re-downloaded every container run** — Fix: mount `./hf_cache` and set `HF_HOME=/workspace/hf_cache`
- 🔴 **Flash Attention warning about GPU init order** — Fix: load model on CPU first, then call `.to('cuda:0')`

---

## Quick Start

```bash
# Clone OpenVLA repo
git clone https://github.com/openvla/openvla.git
cd openvla

# Create data folders
mkdir -p hf_cache datasets checkpoints outputs

# Add Dockerfile, docker-compose.yml, and .env
# (see index.html guide for full file contents)

# Build — takes 30-40 min first time
docker compose build

# Verify RTX 5090 works
docker compose run --rm openvla python -c "
import torch
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device='cuda')
y = torch.mm(x, x)
print('GPU compute test PASSED!')
"

# Load the full 7B model
docker compose run --rm openvla python -c "
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    'openvla/openvla-7b',
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda:0')
print('Model loaded! Parameters:', round(sum(p.numel() for p in vla.parameters()) / 1e9, 2), 'B')
print('GPU memory used:', round(torch.cuda.memory_allocated(0)/1024**3, 2), 'GB')
"
```

Expected output after GPU verify:
```
PyTorch: 2.12.0.dev...+cu128
GPU: NVIDIA GeForce RTX 5090
GPU compute test PASSED!
```

Expected output after model load:
```
Model loaded! Parameters: 7.54 B
GPU memory used: 14.09 GB
```

---

## Verified Results (RTX 5090)

| Check | Result |
|---|---|
| PyTorch version | 2.12.0.dev+cu128 ✓ |
| CUDA version | 12.8 ✓ |
| Compute capability | (12, 0) — sm_120 Blackwell ✓ |
| GPU compute test | PASSED ✓ |
| Model parameters | 7.54B ✓ |
| VRAM used | 14.09 GB / 32 GB ✓ |

---

## Related

- [OpenVLA official repo](https://github.com/openvla/openvla)
- [OpenVLA paper (arXiv)](https://arxiv.org/abs/2406.09246)
- [HuggingFace: openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b)

---

## License

Community documentation. OpenVLA is licensed under MIT (code) — see the [official repo](https://github.com/openvla/openvla) for model licensing details.
