# Environment Setup

## Hardware / System

| Item | Value |
|---|---|
| GPU | NVIDIA Quadro RTX 5000 (16 GB, Turing, sm_75) |
| Driver | 570.133.07 (CUDA 12.8 runtime) |
| OS | Ubuntu 24.04.1 LTS (kernel 6.8.0-41) |
| CPU RAM | 314 GB |
| System compiler | gcc/g++ 13.3.0 |
| System Python | 3.12.3 (not used; conda env below) |

No system `nvcc` and no conda were present; both installed from scratch (2026-07-22).

## Steps

1. **Miniconda** installed to `~/miniconda3` (Miniconda3-latest, conda 26.5.3).
   Note: recent conda requires accepting channel ToS once:
   `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r`
   (the first `conda create` silently failed until this was done).

2. **Env**: `conda create -n scgs python=3.9` (Python 3.9.23, per upstream instructions).

3. **PyTorch**: upstream requirements say `torch==1.12.1+cu113 or any later versions`.
   We use `torch==2.4.1+cu124 / torchvision==0.19.1` (pip, `--index-url https://download.pytorch.org/whl/cu124`).
   Rationale: CUDA 12.4 headers accept host gcc 13 (system compiler); the driver supports CUDA ≤ 12.8.

4. **CUDA toolkit (nvcc)** for building the rasterizer submodules:
   `conda install -n scgs -c "nvidia/label/cuda-12.4.1" cuda-toolkit`.

5. **SC-GS**: `git clone --recursive https://github.com/yihua7/SC-GS`
   (commit pinned in this repo; submodules `diff-gaussian-rasterization` @ d986da0, `simple-knn` @ 44f7642).

6. **Python deps**: `pip install -r requirements.txt` with relaxed pins where the 2022-era pins
   do not build on modern toolchains (deviations recorded below).

7. **pytorch3d**: NOT optional despite being commented out in upstream `requirements.txt` —
   `utils/time_utils.py` (deform network used in training) and the whole editing stack import it
   at module level. Installed from the official prebuilt wheel index if a matching build exists
   (py39 + torch 2.4.1 + cu124), else built from source.

8. **Submodules**: `pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn`
   with `TORCH_CUDA_ARCH_LIST="7.5"`.

## Deviations from upstream instructions

| Upstream | Ours | Reason |
|---|---|---|
| torch 1.12.1+cu113 | torch 2.4.1+cu124 | old CUDA toolchains incompatible with gcc 13 / Ubuntu 24.04; requirements.txt explicitly allows later versions |
| Pillow==7.0.0 | modern Pillow | Pillow 7 predates Python 3.9 and fails to build |
| opencv-python==4.5.5.62 | see final pip freeze | old builds unavailable for the toolchain |
| pytorch3d commented out | installed explicitly | hard import in training + editing code |

## Build errors encountered and fixes

- `conda create` produced no env on first run — cause: unaccepted channel ToS in conda 26.x; fix: `conda tos accept ...` (see step 1).
- `curl` absent on the host — used `wget` instead.
- **diff-gaussian-rasterization failed to compile** with
  `error: namespace "std" has no member "uintptr_t"` / `identifier "uint32_t" is undefined`
  in `cuda_rasterizer/rasterizer_impl.h`. Cause: gcc 13 no longer transitively includes
  `<cstdint>`. Fix: add `#include <cstdint>` and `#include <cstddef>` to `rasterizer_impl.h`
  (committed as a separate patch in the SC-GS working tree). After the patch both submodules
  build cleanly for sm_75. `simple-knn` built without modification.
- pytorch3d prebuilt wheel indexes for py39/torch2.4.x returned HTTP 403 → built
  `pytorch3d==0.7.8` from source (`pip install --no-build-isolation git+...@V0.7.8`,
  `MAX_JOBS=32`, ~40 CPU cores).
- `torch 2.4.1` pulls in `numpy 2.0.2`; downgraded to `numpy==1.26.4` for compatibility with
  the 2022-era SC-GS code (uses deprecated numpy aliases) before building anything against numpy.
