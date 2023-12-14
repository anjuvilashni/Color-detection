import numpy as np
import cv2
import os
import predict as image
from color_detection import detect_color
from simulate_color_blindness import simulate_color
from PIL import Image
from flask import Flask
from flask import render_template, request

app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)
UPLOAD_FOLDER = os.path.join(app_root, "upload")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], "test.jpg")
        f.save(img_path)

        result_classification = image.classify()

        # detect_color(img_path)
        return render_template("index.html", message=result_classification)


@app.route("/detect_color", methods=["GET", "POST"])
def color_detect():
    if request.method == "POST":
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], "test.jpg")
        detect_color(img_path)
        return render_template("index.html")


@app.route("/simulate_color_blindness", methods=["GET", "POST"])
def simulate():
    if request.method == "POST":
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], "test.jpg")
        simulation_type = request.form.get("simulation_type")
        sim_image = simulate_color(img_path, simulation_type)

        return render_template(
            "index.html", sim_image_path="simulate.jpg"
        )


if __name__ == "__main__":
    app.run(debug=True)
