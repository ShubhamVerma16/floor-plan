import pathlib
from demo.image_demo import inference
import os
# img_nm = "newhouse308 Recovered NA.png"
# out_nm = "output.png"

# out_json = 
# print(out_json)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/api_res")



from flask import Flask, request, send_from_directory
import os
from werkzeug.utils import secure_filename
# import inference_ESA_demo_lgsi as p
UPLOAD_FOLDER = 'data/floorplan_point_rend/eval/300-499'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def health():
    return "The server is up and running"

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory(root, f"{pathlib.Path(path).stem}_bound.png")

@app.route("/v1/process_image", methods=["POST"])
def process_image():
    if 'file' not in request.files:
        return "invalid Input"
    file = request.files['file']
    filename = secure_filename(file.filename)
    abs_filename = pathlib.Path(filename).stem
    print("==============")
    print(abs_filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    result = inference(filename, abs_filename)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #print(result)
    return result[0]



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)