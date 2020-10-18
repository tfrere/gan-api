from flask import Flask, jsonify, request

app = Flask(__name__)

def load_model():
    """Load and return the model"""
    # TODO: INSERT CODE
    # return model

# you can then reference this model object in evaluate function/handler
model = load_model()

# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/')
def evaluate():
    """"Preprocessing the data and evaluate the model""""
    # TODO: data/input preprocessing
    # eg: request.files.get('file')
    # eg: request.args.get('style')
    # eg: request.form.get('model_name')

    # TODO: model evaluation
    # eg: prediction = model.eval()

    # TODO: return prediction
    # eg: return jsonify({'score': 0.95})
    return "toto"

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)