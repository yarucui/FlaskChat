import nltk
import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load and preprocess data
with open("intents.json", "r",encoding='utf-8') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
ignore_chars = ['?', '!']
words, classes, documents = [], [], []

# Tokenize and process patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        documents.append((tokenized_words, intent['tag']))
        words.extend(tokenized_words)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Clean up words and classes
words = sorted(set(lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars))
classes = sorted(set(classes))

# Save processed words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Output basic info
print(f"Documents: {len(documents)}")
print(f"Classes: {len(classes)} -> {classes}")
print(f"Words: {len(words)} -> {words}")

# Prepare training data
training_data = []
output_empty = [0] * len(classes)

for pattern_words, tag in documents:
    bag = [1 if lemmatizer.lemmatize(word.lower()) in pattern_words else 0 for word in words]
    output = output_empty[:]
    output[classes.index(tag)] = 1
    training_data.append((bag, output))

random.shuffle(training_data)
X_train = np.array([data[0] for data in training_data])
Y_train = np.array([data[1] for data in training_data])

print("Training data created")

# Define and compile model
model = Sequential([
    Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(Y_train[0]), activation='softmax')
])

optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save("model.h5")
print("Model has been trained and saved.")
