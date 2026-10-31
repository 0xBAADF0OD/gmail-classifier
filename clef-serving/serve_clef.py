import os
import sys

from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download

model = os.environ.get("CLEF_MODEL", "simonlehmann/clef-NVFP4")
MODEL_DIR = model if os.path.isdir(model) else snapshot_download(model)
sys.path.insert(0, MODEL_DIR)

API_KEYS = {k for k in os.environ.get("CLEF_API_KEYS", "").split(",") if k}

app = FastAPI()

from clef_vllm import ClefVLLM

clef = ClefVLLM(
    MODEL_DIR,
    max_model_len=int(os.environ.get("CLEF_MAX_TOKENS", "16384")),
    gpu_memory_utilization=float(os.environ.get("CLEF_GPU_MEM", "0.30")),
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "clef",
                "object": "model",
                "owned_by": "cloudflare",
                "release_date": os.environ.get("CLEF_RELEASE_DATE", "2026-10-01"),
            }
        ],
    }


@app.post("/v1/systemone")
async def systemone(req: Request):
    if API_KEYS and req.headers.get("authorization", "").removeprefix("Bearer ").strip() not in API_KEYS:
        raise HTTPException(401)
    body = await req.json()
    body.setdefault("model", "clef")
    try:
        return clef.systemone(body)
    except Exception as e:
        raise HTTPException(502, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("CLEF_PORT", "8080")))
