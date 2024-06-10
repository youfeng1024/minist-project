from flask import Flask, jsonify, request
from model import *
import PIL.Image as Image
from script.inference import inference

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if 'model_name' not in request.form:
        return jsonify({"error": "No model name specified"}), 400

    model_name = request.form['model_name']

    if file:
        image = Image.open(file.stream)

        result = inference(image=image, model_name=model_name)
        return result

if __name__ == '__main__':
    app.run(debug=True)


