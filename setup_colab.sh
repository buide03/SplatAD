#!/usr/bin/env bash
set -Eeuo pipefail

stage="GPU detection"
trap 'status=$?; printf "\nTask 6 setup failed during %s (line %s, exit %s).\n" "$stage" "$LINENO" "$status" >&2; exit "$status"' ERR

capability=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader,nounits)
case "$capability" in
  7.0|7.5|8.0|8.6|8.9|9.0) ;;
  *) printf 'Choose a CUDA 11.8 compatible GPU such as T4 or A100 (detected %s).\n' "$capability" >&2; exit 1 ;;
esac
export TORCH_CUDA_ARCH_LIST="$capability"
export TCNN_CUDA_ARCHITECTURES="${capability/./}"
printf 'Building for CUDA compute capability %s\n' "$capability"

root=/content/splatad-runtime
env="$root/env"
stage="Python and CUDA toolchain"
mkdir -p "$root"
if [[ ! -x "$root/bin/micromamba" ]]; then
  curl -fsSL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$root" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$root/mamba"
mamba="$root/bin/micromamba"
if [[ ! -x "$env/bin/python" ]]; then
  "$mamba" create -y -p "$env" -c conda-forge -c nvidia/label/cuda-11.8.0 \
    python=3.10 pip cuda-toolkit=11.8 gxx_linux-64=11 ffmpeg
fi
export PATH="$env/bin:$PATH"
export CUDA_HOME="$env"
export LD_LIBRARY_PATH="$env/lib:${LD_LIBRARY_PATH:-}"
export CC="$env/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$env/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=8
"$env/bin/python" --version
"$env/bin/nvcc" --version
"$CXX" --version

neurad=aea389a936890eb48e1b43a493a52c0194559b48
splatad=6e31ad766d39e0c33f9034a2ed772d51364b2343
tcnn=8e6e242f36dd197134c9b9275a8e5108a8e3af78
ready="$root/ready-$capability"
if [[ ! -f "$ready" ]]; then
  stage="PyTorch and torchvision wheels"
  echo "Installing PyTorch 2.0.1 and torchvision 0.15.2 for CUDA 11.8"
  python -m pip install 'setuptools<70' wheel ninja pybind11 tomli
  python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  python -c 'import torch, torchvision; assert torch.version.cuda == "11.8"; print(torch.__version__, torchvision.__version__)'
  cat > "$root/constraints.txt" <<'EOF'
numpy==1.26.4
pandas==2.2.3
dataclass-wizard==0.22.3
torch==2.0.1+cu118
torchvision==0.15.2+cu118
EOF
  stage="NeuRAD Studio source"
  if [[ ! -d "$root/neurad-studio/.git" ]]; then
    git clone https://github.com/georghess/neurad-studio.git "$root/neurad-studio"
  fi
  git -C "$root/neurad-studio" checkout --detach "$neurad"
  # Install the declared dependencies, substituting the pinned gsplat revision.
  stage="tiny-cuda-nn build"
  echo "Building tiny-cuda-nn for compute capability $capability"
  python -m pip install --no-build-isolation -c "$root/constraints.txt" \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@$tcnn#subdirectory=bindings/torch"
  stage="SplatAD gsplat package"
  BUILD_CUDA=0 python -m pip install --no-build-isolation -c "$root/constraints.txt" \
    "gsplat @ git+https://github.com/carlinds/splatad.git@$splatad"
  python - "$root/neurad-studio/pyproject.toml" > "$root/neurad-requirements.txt" <<'PY'
import sys
import tomli
with open(sys.argv[1], 'rb') as source:
    for requirement in tomli.load(source)['project']['dependencies']:
        if not requirement.startswith('gsplat@'):
            print(requirement)
PY
  stage="Nerfstudio dependencies"
  python -m pip install -c "$root/constraints.txt" -r "$root/neurad-requirements.txt" 'dataclass-wizard<0.23'
  python -m pip install --no-build-isolation --no-deps -e "$root/neurad-studio"
fi
stage="CUDA and gsplat verification"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot access the Colab GPU"
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
import tinycudann
from gsplat.cuda._backend import _C
from gsplat.rendering import rasterization, lidar_rasterization
assert _C is not None, "gsplat did not load its CUDA extension"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
stage="Nerfstudio CLI verification"
ns-train splatad pandaset-data --help > "$root/train-help.txt"
ns-render dataset --help > "$root/render-help.txt"
python -m pip freeze > "$root/environment.txt"
printf 'neurad-studio %s\ngsplat %s\ntiny-cuda-nn %s\n' "$neurad" "$splatad" "$tcnn" > "$root/commits.txt"
touch "$ready"
