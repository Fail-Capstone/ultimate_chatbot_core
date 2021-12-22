import pickle
from os import error
from text_preprocess import text_preprocess
from db_connect import get_collection

def get_data_server():
    try:
        intents = get_collection('intents').find()
        data_train = []
        answer = []
        for intent in intents:
            for pattern in intent['patterns']:
                pattern = text_preprocess(pattern)
                data_train.append({"Question": pattern, "Intent": intent['tag']})
            answer.append({"tag": intent['tag'], "response": intent['response']})
        pickle.dump(answer, open('answer.pkl', 'wb'))
        return data_train
    except error:
        print(error)

