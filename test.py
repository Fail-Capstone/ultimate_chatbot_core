# # from gensim.models import KeyedVectors
# from sklearn import preprocessing
# from tensorflow.keras import optimizers
# from tensorflow.python.keras.engine import input_layer
# from text_preprocess import text_preprocess
# import numpy as np
# import json
# import pandas as pd
# from sklearn import svm
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.keras.layers import *
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Model
# from keras.layers import Dense, LSTM
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical

# with open('intent1.json', encoding='utf8') as file:
#     data = json.loads(file.read())

# patterns = []
# tags = []
# answer = []
# for intent in data:
#     for pattern in intent['patterns']:
#         patterns.append(text_preprocess(pattern))
#         tags.append(intent['tag'])
#     answer.append(intent['response'])

# def truncated(data, n_components=300):
#     svd = TruncatedSVD(n_components=n_components, random_state=42)
#     svd.fit(data)
#     return svd.transform(data)

# # model = KeyedVectors.load_word2vec_format("./baomoi.vn.model.bin", binary=True, unicode_errors='ignore')
# # vocab = model.key_to_index
# # wv = model
# # print(wv)
# # def get_word2vec_data(X):
# #     word2vec_data = []
# #     for x in X:
# #         sentence = []
# #         for word in x.split(" "):
# #             if word in vocab:
# #               sentence=sentence+wv[word].ravel().tolist()
# #         word2vec_data.append(sentence)
# #     return word2vec_data

# # def change_to_word2vec(data):
# #   data2vec=get_word2vec_data(data)
# #   lengthOfdata=[len(data2vec[i]) for i,n in enumerate(data2vec)]
# #   for i,n in enumerate(data):
# #     if(len(data2vec[i])<max(lengthOfdata)):
# #       for j in range(1,(max(lengthOfdata)-len(data2vec[i]))+1):
# #         data2vec[i].append(0)
# #   return truncated(np.array(data2vec))
# # X_data_w2v_patterns=change_to_word2vec(patterns)

# X_train, X_test, y_train, y_test = train_test_split(patterns, tags, test_size=0.1, random_state=42)

# tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_features=10000)
# tfidf_vect.fit(X_train)
# X_train_tfidf = tfidf_vect.transform(X_train)
# X_test_tfidf = tfidf_vect.transform(X_test)

# svd = TruncatedSVD(n_components=300, random_state=42)
# svd.fit(X_train_tfidf)
# X_train_tfidf_svd = svd.transform(X_train_tfidf)
# X_test_tfidf_svd = svd.transform(X_test_tfidf)

# # def lstm():
# #   input_layer = Input(shape=(300,))
# #   layer = Reshape((1,300))(input_layer)
# #   layer = LSTM(256, activation='relu', dropout=0.4, recurrent_dropout=0.3)(layer)
# #   layer = Dense(512, activation='relu')(layer)
# #   layer = Dense(256, activation='relu')(layer)
# #   layer = Dense(128, activation='relu')(layer)
# #   outout_layer = Dense(10, activation='softmax')(layer)
# #   model = Model(input_layer,outout_layer)
# #   model.compile(optimizer='Adam', loss='crossentropy', metrics=['accuracy'])
# #   return model

# # model = lstm()
# # model.summary()

# encoder = LabelEncoder()
# y_train_encoder = encoder.fit_transform(y_train)
# y_test_encoder = encoder.fit_transform(y_test)

# X_train_model, X_val_model, y_train_model, y_val_model = train_test_split(X_train_tfidf, y_train_encoder, test_size=0.1, random_state=42, stratify=y_train_encoder)
# # model.fit(X_train_model, y_train_model, validation_data=(X_val_model, y_val_model), epochs=100, batch_size=64)

# # model = LogisticRegression()
# # model.fit(X_train_model, y_train_model)

# # train_predictions = model.predict(X_train_model)
# # val_predictions = model.predict(X_val_model)
# # test_predictions = model.predict(X_test_tfidf)

# # val_predictions = val_predictions.argmax(axis=-1)
# # test_predictions = test_predictions.argmax(axis=-1)
# # train_predictions = train_predictions.argmax(axis=-1)

# # print("Train accuracy", accuracy_score(train_predictions, y_train_model))
# # print("Validation accuracy: ", accuracy_score(val_predictions, y_val_model))
# # print("Test accuracy: ", accuracy_score(test_predictions, y_test_encoder))

# tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=10000)
        # param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        # grid = GridSearchCV(SVC(), param_grid) 
        # grid.fit(X_train, y_train)
        # print(grid.best_params_)
        # print(grid.best_estimator_)
        # grid_predictions = grid.predict(X_test)
        # print(classification_report(y_test, grid_predictions)) 

        # logistic_model.clf.fit(X_train,y_train)
        # predictions = logistic_model.clf.predict(X_test)
        # print(classification_report(y_test,predictions))
        # svm_model.clf.fit(X_train,y_train)
        # predictions1 = svm_model.clf.predict(X_test)
        # print(classification_report(y_test, predictions1))
        # naivie_model = MultinomialNB_Model()
        # naivie_model.clf.fit(X_train, y_train)
        # predictions2=naivie_model.clf.predict(X_test)
        # print(classification_report(y_test, predictions2))
        # X_train, X_test, y_train, y_test = train_test_split(df_train["Question"], df_train['Intent'])