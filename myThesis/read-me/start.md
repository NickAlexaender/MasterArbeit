
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



# Run it all:

python myThesis/conceptualize_image.py

- 

python myThesis/encoder/nd_on_transformer_encoder.py

python myThesis/encoder/calculate_IoU_for_encoder.py

-

python myThesis/decoder/nd_on_transformer_decoder.py

python myThesis/decoder/calculate_IoU_for_decoder.py

- 

python myThesis/lrp/get_neurons_to_look_at.py