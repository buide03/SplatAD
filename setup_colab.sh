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
printf 'Building for CUDA compute capability %s\n' "$capability"

root=/content/splatad-runtime
env="$root/env"
cuda="$root/cuda-11.8"
stage="Python and CUDA toolchain"
mkdir -p "$root"
if [[ ! -x "$root/bin/micromamba" ]]; then
  curl -fsSL --connect-timeout 15 --max-time 120 --retry 3 \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$root" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$root/mamba"
mamba="$root/bin/micromamba"
if [[ ! -x "$env/bin/python" ]]; then
  "$mamba" create -y -p "$env" --override-channels -c conda-forge \
    python=3.10 pip gxx_linux-64=11 ffmpeg
fi
stage="NVIDIA CUDA 11.8 compiler and headers"
# The older toolkit metapackage has unbounded dependencies. Isolate its release channel.
if [[ ! -x "$cuda/bin/nvcc" || ! -f "$cuda/include/cuda_runtime_api.h" ]]; then
  "$mamba" create -y -p "$cuda" --override-channels -c nvidia/label/cuda-11.8.0 \
    cuda-nvcc=11.8.89 cuda-libraries-dev=11.8.0
fi
export PATH="$cuda/bin:$env/bin:$PATH"
export CUDA_HOME="$cuda"
export LD_LIBRARY_PATH="$cuda/lib:$env/lib:${LD_LIBRARY_PATH:-}"
export CC="$env/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$env/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=8
"$env/bin/python" --version
nvcc_version=$("$cuda/bin/nvcc" --version)
printf '%s\n' "$nvcc_version"
if [[ "$nvcc_version" != *"release 11.8,"* || ! -f "$cuda/include/cuda_runtime_api.h" ]]; then
  printf 'Expected CUDA 11.8 nvcc and cuda_runtime_api.h under %s.\n' "$cuda" >&2
  exit 1
fi
"$CXX" --version
stage="CUDA compiler and linker preflight"
printf '#include <cuda_runtime_api.h>\nint main() { int n = 0; return cudaGetDeviceCount(&n); }\n' |
  "$CXX" -x c++ -I"$cuda/include" - -L"$cuda/lib" -lcudart -o "$root/cuda-preflight"
"$root/cuda-preflight"

neurad=aea389a936890eb48e1b43a493a52c0194559b48
splatad=6e31ad766d39e0c33f9034a2ed772d51364b2343
dependencies_ready="$root/dependencies-ready-cu118"
if [[ ! -f "$dependencies_ready" ]]; then
  cat > "$root/constraints.txt" <<'EOF'
numpy==1.26.4
pandas==2.2.3
dataclass-wizard==0.22.3
torch==2.0.1+cu118
torchvision==0.15.2+cu118
EOF
  stage="PyTorch, torchvision and NumPy"
  echo "Installing PyTorch 2.0.1, torchvision 0.15.2 and NumPy 1.26 for CUDA 11.8"
  python -m pip install 'setuptools<70' wheel ninja pybind11 tomli
  python -m pip install -c "$root/constraints.txt" numpy torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu118
  python -c 'import numpy, torch, torchvision; torch.from_numpy(numpy.zeros(1)); assert torch.version.cuda == "11.8"; print(torch.__version__, torchvision.__version__, numpy.__version__)'
  stage="NeuRAD Studio source"
  if [[ ! -d "$root/neurad-studio/.git" ]]; then
    git clone https://github.com/georghess/neurad-studio.git "$root/neurad-studio"
  fi
  git -C "$root/neurad-studio" checkout --detach "$neurad"
  # Install the declared dependencies, substituting the pinned gsplat revision.
  # SplatAD's LiDAR decoder uses its supported PyTorch MLP in colab_pipeline.py.
  stage="SplatAD gsplat package"
  BUILD_CUDA=0 python -m pip install --no-build-isolation -c "$root/constraints.txt" \
    "gsplat @ git+https://github.com/carlinds/splatad.git@$splatad"
  python - "$root/neurad-studio/pyproject.toml" > "$root/neurad-requirements.txt" <<'PY'
import sys
import tomli
revisions = {
    'viser': 'viser @ git+https://github.com/atonderski/viser.git@57142e42df8edd4de33fd60a08d6bb6c35970aa1',
    'pandaset': 'pandaset @ git+https://github.com/scaleapi/pandaset-devkit.git@59be180e2a3f3e37f6d66af9e67bf944ccbf6ec0#subdirectory=python',
}
with open(sys.argv[1], 'rb') as source:
    for requirement in tomli.load(source)['project']['dependencies']:
        package = requirement.split('@', 1)[0].strip()
        if package != 'gsplat':
            print(revisions.get(package, requirement))
PY
  stage="Nerfstudio dependencies"
  python -m pip install -c "$root/constraints.txt" -r "$root/neurad-requirements.txt" 'dataclass-wizard<0.23'
  python -m pip install --no-build-isolation --no-deps -e "$root/neurad-studio"
fi
stage="Nerfstudio CLI verification"
ns-train splatad pandaset-data --help > "$root/train-help.txt"
ns-render dataset --help > "$root/render-help.txt"
touch "$dependencies_ready"

stage="CUDA and gsplat verification"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot access the Colab GPU"
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
from gsplat.cuda._backend import _C
from gsplat.rendering import rasterization, lidar_rasterization
assert _C is not None, "gsplat did not load its CUDA extension"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
python -m pip freeze > "$root/environment.txt"
printf 'neurad-studio %s\ngsplat %s\n' "$neurad" "$splatad" > "$root/commits.txt"
printf '%s\n' "$capability" > "$root/cuda-arch.txt"
