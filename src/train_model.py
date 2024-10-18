# src/train_model.py
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint
import numpy as np

def train_model():
    """Train the LSTM model with the pre-processed sequences."""
    # Load processed data
    input_sequences = np.load('data/processed/input_sequences.npy')
    output_notes = np.load('data/processed/output_notes.npy')
    n_vocab = output_notes.shape[1]  # Vocabulary size is number of output categories

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(512, input_shape=(input_sequences.shape[1], input_sequences.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Define a checkpoint to save the best model during training
    checkpoint = ModelCheckpoint(
        "model/music_generator_model.h5",  # Path to save model
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]

    # Train the model
    model.fit(input_sequences, output_notes, epochs=100, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_model()
