sudo apt-get update
sudo apt-get install --no-install-recommends -y git-all ninja-build python3.10 python3-pip python3-venv 
sudo apt-get install --no-install-recommends -y awscli build-essential curl wget git-lfs ffmpeg libsm6 libxext6 poppler-utils
sudo apt-get install --no-install-recommends -y tesseract-ocr nano systemctl
sudo apt-get install -y redis-server
sudo apt-get autoremove
python3 -m venv ~/mpilo-env
