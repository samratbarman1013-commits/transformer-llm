# Transformer API server
#
# Build:   docker build -t transformer-api .
# Run:     docker run -p 8000:8000 \
#            -e TRANSFORMER_API_KEY=<your-secret> \
#            -v $(pwd)/server-data:/app/server-data \
#            transformer-api
#
# server-data/ must contain the PRIVATE model files:
#   model_best.pt   (trained weights from the CI chain)
#   tokenizer.json  (the 32k BPE tokenizer)
# Never commit server-data/ — it is gitignored.

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY transformer/ transformer/
COPY server/ server/

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
