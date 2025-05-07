from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
from io import BytesIO
from PIL import Image

app = FastAPI()

# Izinkan frontend mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image/")
async def process_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=25
    )

    result = {
        "peserta": pytesseract.image_to_string(gray[:200, :600]),
        "nilai_dilingkari": [],
    }

    if circles is not None:
        for (x, y, r) in np.round(circles[0, :]).astype("int"):
            cropped = gray[y-15:y+15, x-15:x+15]
            nilai = pytesseract.image_to_string(cropped, config="--psm 6 digits")
            result["nilai_dilingkari"].append({
                "x": int(x),
                "y": int(y),
                "nilai": nilai.strip()
            })

    return result
