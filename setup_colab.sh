#!/usr/bin/env bash
set -euo pipefail

root=/content/splatad-runtime
env="$root/env"
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

capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
case "$capability" in
  7.0|7.5|8.0|8.6|8.9|9.0) ;;
  *) echo "CUDA 11.8 recipe needs a supported GPU, such as Colab T4, L4 or A100 (found $capability)." >&2; exit 1 ;;
esac
export TORCH_CUDA_ARCH_LIST="$capability"
export TCNN_CUDA_ARCHITECTURES="${capability/./}"

neurad=aea389a936890eb48e1b43a493a52c0194559b48
splatad=6e31ad766d39e0c33f9034a2ed772d51364b2343
tcnn=8e6e242f36dd197134c9b9275a8e5108a8e3af78
if [[ ! -f "$root/ready" ]]; then
  python -m pip install 'setuptools<70' wheel ninja pybind11 tomli
  python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  cat > "$root/constraints.txt" <<'EOF'
numpy==1.26.4
pandas==2.2.3
dataclass-wizard==0.22.3
torch==2.0.1+cu118
torchvision==0.15.2+cu118
EOF
  if [[ ! -d "$root/neurad-studio/.git" ]]; then
    git clone https://github.com/georghess/neurad-studio.git "$root/neurad-studio"
  fi
  git -C "$root/neurad-studio" checkout --detach "$neurad"
  # Install the declared dependencies, substituting the pinned gsplat revision.
  python -m pip install --no-build-isolation -c "$root/constraints.txt" \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@$tcnn#subdirectory=bindings/torch"
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
  python -m pip install -c "$root/constraints.txt" -r "$root/neurad-requirements.txt" 'dataclass-wizard<0.23'
  python -m pip install --no-build-isolation --no-deps -e "$root/neurad-studio"
fi
python -c 'import torch; from gsplat.rendering import rasterization, lidar_rasterization; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
ns-train splatad pandaset-data --help > "$root/train-help.txt"
ns-render dataset --help > "$root/render-help.txt"
python -m pip freeze > "$root/environment.txt"
printf 'neurad-studio %s\ngsplat %s\ntiny-cuda-nn %s\n' "$neurad" "$splatad" "$tcnn" > "$root/commits.txt"
touch "$root/ready"
