FROM waggle/plugin-base:1.1.1-ml-dev

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ /app/

ENTRYPOINT ["python3", "main.py"]
