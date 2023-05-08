import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import glob
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
IMAGE_DISPLAY_FOLDER = "static/images"
MODELS_FOLDER = "models"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["IMAGE_DISPLAY_FOLDER"] = IMAGE_DISPLAY_FOLDER
app.config["MODELS_FOLDER"] = MODELS_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    models_list = [os.path.basename(model_path)
                   for model_path in glob.glob("models/*.h5")]
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # if folder not exist, create it
            if not os.path.exists(app.config["UPLOAD_FOLDER"]):
                os.mkdir(app.config["UPLOAD_FOLDER"])
            if not os.path.exists(app.config["IMAGE_DISPLAY_FOLDER"]):
                os.mkdir('static')
                os.mkdir(app.config["IMAGE_DISPLAY_FOLDER"])
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            shutil.copy(file_path, os.path.join(
                app.config["IMAGE_DISPLAY_FOLDER"], file.filename))
            selected_model = request.form["model"]
            model_path = os.path.join("models", selected_model)
            predicted_class, predicted_probability = predict(
                file_path, model_path)
            return render_template("result.html", predicted_class=predicted_class, predicted_probability=predicted_probability, image_file=file.filename)
    return render_template("index.html", models_list=models_list)


def predict(file_path, model_path):
    model = load_model(model_path)
    img = image.load_img(file_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.  # Normaliser les valeurs des pixels

    predictions = model.predict(x)
    predicted_index = np.argmax(predictions[0])

    test_generator = {'adidas': 0, 'converse': 1, 'nike': 2}
    predicted_class = list(test_generator.keys())[list(
        test_generator.values()).index(predicted_index)]
    predicted_probability = predictions[0][predicted_index]
    return predicted_class, predicted_probability


@app.route("/upload_model", methods=["GET", "POST"])
def upload_model():
    try:
        if request.method == "POST":
            if "file" not in request.files:
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)
            if file and file.filename.endswith('.h5'):
                # if folder not exist, create it
                if not os.path.exists(app.config["MODELS_FOLDER"]):
                    os.mkdir(app.config["MODELS_FOLDER"])
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["MODELS_FOLDER"], filename)
                file.save(file_path)
                return redirect(url_for('index'))
        return render_template("upload.html")

    except Exception as e:
        app.logger.error("Erreur lors de l'upload du modèle : {}".format(e))
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Une erreur est survenue lors de l'upload du modèle"}), 500


if __name__ == "__main__":
    app.run(debug=True)
