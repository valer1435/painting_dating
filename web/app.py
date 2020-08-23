from flask import Flask, render_template, request, jsonify
from predict_period import make_prediction
app = Flask(__name__)

import torch
from PIL import Image

@app.route("/")
def index():
    return render_template("main.html")

@app.route('/upload', methods=['POST'])
def upload_file():
     if request.method == 'POST':
        f = request.files['file']
        f.filename = "img.jpg"
        f.save(f.filename)
#
#        predict_period(f, device)
        img = Image.open("img.jpg")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(make_prediction(img, device)[0])
        data = {'age': make_prediction(img, device)[0]}
        return jsonify(data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5011))
    app.run(host='0.0.0.0', port=port)

