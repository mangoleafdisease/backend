

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/7")

CLASS_NAMES = ["Anthracnose", "Bacterial Canker", "Black Soothy Mold", "Cutting Weevil", "Die Back", "Gail Midge", "Healthy", "Powdery Mildew", "Sooty Mould" ]
RECOMMENDATIONS = None
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]

    confidence = np.max(predictions[0])
    conf = int(float(confidence) * 100)
    if conf < 90:
        return {
            "unable": True,
            "class": predicted_class,
            'confidence': float(confidence)
        }
    elif conf == 100 and predicted_class == "Black Soothy Mold":
        return {
            "unable": True,
            "class": predicted_class,
            'confidence': float(confidence)
        }
    else:
        return {
        'class': predicted_class,
        'confidence': float(confidence)
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

