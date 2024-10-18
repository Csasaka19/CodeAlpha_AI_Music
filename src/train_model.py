import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
import pickle


with open("data/processed/notes.pkl", "rb") as filepath:
    notes = pickle.load(filepath)

# Get all the pitch names and unique note count
n_vocab = len(set(notes))

# Create sequences of notes (input) and the following note (output)
sequence_length = 100
input_sequences = []
output_notes = []

# Mapping notes to integers for easy processing
note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    input_sequences.append([note_to_int[note] for note in sequence_in])
    output_notes.append(note_to_int[sequence_out])

# Reshape input sequences for the LSTM
input_sequences = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1))
input_sequences = input_sequences / float(n_vocab) 

# One-hot encode the output
output_notes = np.eye(n_vocab)[output_notes]

# Build the RNN model
model = Sequential()
model.add(LSTM(512, input_shape=(input_sequences.shape[1], input_sequences.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(n_vocab, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(input_sequences, output_notes, epochs=100, batch_size=64)

# Save the trained model
model.save("model/music_generator_model.h5")
print("Model trained and saved to 'model/music_generator_model.h5'")
