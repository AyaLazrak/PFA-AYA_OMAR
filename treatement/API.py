from flask import Flask, request, jsonify, render_template
from Detection import FraudDetectionAI

app = Flask(__name__)
fraud_detection = None


@app.route('/')
def index():
    return render_template('./WEB/Home.html')

@app.route('/load_model', methods=['GET'])
def load_model():
    print(request.method)
    global fraud_detection
    data_file = request.args.get('filePath')
    fraud_detection = FraudDetectionAI(data_file)
    fraud_detection.train()
    return "Model loaded successfully"

@app.route('/predict', methods=["POST", "GET"])
def predict():
    data = request.get_json()
    predictions = fraud_detection.predict(data)
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
