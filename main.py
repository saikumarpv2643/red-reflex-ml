from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from inference import predict

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ML server running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result, confidence = predict(image)

    return {
        "status": "success",
        "result": result,
        "confidence": round(confidence, 3)
    }
