

import random
import json
import pickle
import numpy as np

#Used for contextualization and Natural Language Process (NLP) Tasks
import nltk
from nltk.stem import WordNetLemmatizer

#Using Tensor flow model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

#Lemmatizer is used for grouping together different words to analyze it as a sentence
lemmatizer = WordNetLemmatizer()

#Used to load the intents.json file
intents = json.loads(open('intents.json').read())

#Processing the intents
words = []
classes = []
documents = []
ignore_letters = ['?', '!',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenizes each word in a sentence
        word_list = nltk.word_tokenize(pattern)

        #add to words list
        words.extend(word_list)

        #add to documents in our corpus
        documents.append((word_list,intent['tag']))

        #add to our classes List
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#analyzing the words using lemmatizer and sorting them
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

#pickling the responses and temporarily dumping them to the documents
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Process for training the data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag =[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('Done')