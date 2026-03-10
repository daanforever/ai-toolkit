
call .\venv\Scripts\activate

pip install --no-cache-dir torch==2.9.0+cu130 torchvision==0.24.0+cu130 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt