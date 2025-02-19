FROM waggle/plugin-opencv:4.1.1

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --upgrade setuptools pip wheel
RUN pip3 install nvidia-pyindex
RUN pip3 install nvidia-cuda-runtime-cu11
# RUN pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

COPY app/ /app/
# CMD ["tail", "-f", "/dev/null"]
ENTRYPOINT ["python3", "main.py"]
