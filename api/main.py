from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.saving.save import load_model
from keras.preprocessing.image import ImageDataGenerator, image
from numpy.lib.npyio import load
from numpy.lib.ufunclike import fix
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt




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

#MODEL = tf.keras.models.load_model('../cat_dog_model.h5')
model = load_model('../model_backup/v2.3_batch=128, val_batch=256, epoch=11, acc=0.90, loss=0.25, val_loss=0.33/cat_dog_model.h5')

CLASS_NAMES = ["Cat", "Dog"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    return img


@app.post("/predict")

async def predict(
    file: UploadFile = File(...)
):
    # print(file)
    img = read_file_as_image(await file.read())
    PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
    
    PIL_image = PIL_image.resize((150, 150))
    # PIL_image.show()

    #img = Image.open(file)
    #img.show()
    #PIL_image = img.convert("L")
    #img_batch = np.expand_dims(img, 0)
    #prediction = model.predict(img_batch)
    
    # img_path = 'C:/Users/Richie Lee/Desktop/College/1 San Jose State University/9 2021 fall semester/CMPE 188/project/CNN_cat_dog_CN/data/test/dog/dog.4073.jpg'
    # img = image.load_img(img_path, target_size=(150, 150))
    # img.show()
    img_tensor = image.img_to_array(PIL_image)
    img_tensor = img_tensor / 255
    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model.predict(img_tensor)

    

    #predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    if prediction[0] > 0.5:
        predict_class = CLASS_NAMES[1]
        confidence = np.max(prediction[0])
    else:
        predict_class = CLASS_NAMES[0]
        confidence = 1 - np.max(prediction[0])
    return {
        'class': predict_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)