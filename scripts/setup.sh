# conda create -n srmgn pip
# conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda activate srmgn
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install cupy-cuda11x
pip install opencv-python
pip install pytorch-fid
conda install -c conda-forge tqdm
