import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# ==========================
# FOLDER
# ==========================

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"
MODEL_NAME = "vgg16_ikan.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================
# DOWNLOAD MODEL
# ==========================

FILE_ID = "1e11I7POHIdZ5v3r_P0j5tz4FD61dKDhd"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(url, MODEL_PATH, quiet=False)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================
# LOAD CLASS NAME
# ==========================

DATASET_PATH = "dataset/train"

class_names = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Class :", class_names)

# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# PREDICT
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img = img.resize((224,224))

    img = np.array(img,dtype=np.float32)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    fish = class_names[index]

    confidence = float(np.max(prediction))*100

    return render_template(
        "index.html",
        prediction=fish,
        confidence=round(confidence,2),
        filename=file.filename
    )


# ==========================
# SHOW IMAGE
# ==========================

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=True)