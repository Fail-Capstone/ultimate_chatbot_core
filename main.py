from crypt import methods
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import get_answer
from conversation_preprocess import conversration_predict
from model import train
from model_cv import train_model_conversation
# Init app
app = Flask(__name__)

# Flask cors
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=['POST'])
async def receiveAnswer():
    try:
        data = request.get_json()
        question = data['question']
        answer = get_answer(question)
        return answer
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict-conversation", methods=["POST"])
async def predictConversation():
    try:
        data=request.get_json()
        conversation = data['conversation']
        conversration_predict(conversation)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=['GET'])
async def home():
    return 'Đây là home'

@app.route("/train", methods=['GET'])
async def trainModel():
    return train()

@app.route("/")
async def trainModelConversation():
    return train_model_conversation()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

