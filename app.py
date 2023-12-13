import numpy as np
import cv2
import os
from flask import Flask
from flask import (
    render_template,
    request
)

app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def success():
    if request.method == "POST":
        f = request.files["file"]
        f.save(f.filename)
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
