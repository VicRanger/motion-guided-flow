conda create --name study-py311-cu124 python=3.11 --yes
conda activate study-py311-cu124
conda update -n base -c defaults conda --yes
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install matplotlib --yes
conda install -c conda-forge tqdm --yes
conda install -c conda-forge scikit-image --yes
@REM conda install -c conda-forge tensorboard --yes
pip install tb-nightly 
conda install pillow --yes
conda install lxml --yes
pip install opencv-python
pip install onnx
pip install onnxsim
pip install pytorch-msssim
pip install lpips
pip install yacs
pip install timm
@REM choose your cuda version (cupy-cuda12x or cupy-cuda11x)
pip install cupy-cuda12x
@REM optional: install ipykernel
@REM pip install ipykernel
@REM conda install -n study-py311-cu117 ipykernel
@REM python -m ipykernel install --user --name study-py311-cu117 --display-name "study-py311-cu117"
