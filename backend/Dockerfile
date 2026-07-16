FROM python:3.11-slim

WORKDIR /app
# HF Spaces: only /tmp is writable; keep caches there
ENV HF_HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
