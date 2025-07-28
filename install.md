## Folgendes muss vorher Stimmen:
python -c "import platform; print(platform.machine())"
-> arm64



### Um es zu Installieren auf Mac folge den folgenden Anweisungen

conda create -n maskdino-arm python=3.10 -y
conda activate maskdino-arm

pip install torch torchvision torchaudio
pip install cython pycocotools matplotlib opencv-python tqdm yacs termcolor cloudpickle tensorboard
  
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout v0.6

CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" FORCE_CUDA=0 \
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt

pip uninstall opencv-python
pip install opencv-python==4.7.0.72
pip install "numpy<2" 

cd maskdino/modeling/pixel_decoder/ops

# Verändere die Datei
if CUDA_HOME is None:
    raise NotImplementedError('CUDA_HOME is None. Please set environment variable CUDA_HOME.')
# zu ...
if CUDA_HOME is None:
    print("Warning: CUDA_HOME is None. Proceeding with CPU-only build.")

# und /Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/maskdino/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.cpp
#include <ATen/cuda/CUDAContext.h>
# zu
// #include <ATen/cuda/CUDAContext.h>


# CUDA-Free Kompilierung
export CUDA_HOME=""
export FORCE_CUDA=0
python setup.py build_ext --inplace -j 4


## # Erstelle die Gewichte: (in MaskDINO)
mkdir -p weights
# Lade die Gewichte von https://github.com/IDEA-Research/MaskDINO/blob/main/README.md -> MaskDINO (hid 1024)
# lade sie in die weights Datei



# Zusätzlich manchmal nötig
  pip install Pillow
  pip install torch torchvision torchaudio