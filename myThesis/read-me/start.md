
# Zuerst einfach versuchen mir

conda activate maskdino-arm

# Falls das nicht klappt

conda create -n maskdino-arm python=3.10 -y

pip install torch torchvision torchaudio
pip install cython pycocotools matplotlib opencv-python tqdm yacs termcolor cloudpickle tensorboard

CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" FORCE_CUDA=0 \
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

cd MaskDINO
pip install -r requirements.txt