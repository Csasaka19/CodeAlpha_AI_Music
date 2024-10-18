# src/generate_music.py
from keras.models import load_model
import numpy as np
from music21 import stream, note, chord

def generate_music():
    """Generate new music using the trained LSTM model."""
    # Load the trained model
    model = load_model('model/music_generator_model.h5')

    # Load the note dictionary and sequences
    input_sequences = np.load('data/processed/input_sequences.npy')
    with open('data/processed/notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitch_names = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
    n_vocab = len(pitch_names)

    # Randomly pick a sequence from input to start
    start = np.random.randint(0, len(input_sequences) - 1)
    pattern = input_sequences[start]
    generated_notes = []

    # Generate 500 notes
    for i in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        generated_notes.append(result)

        # Append the prediction to the input and remove the first value
        pattern = np.append(pattern, index)
        pattern = pattern[1:]

    # Convert the output notes to MIDI
    offset = 0
    output_notes = []

    for pattern in generated_notes:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    # Create a MIDI stream and write to file
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output/generated_music.mid')

if __name__ == '__main__':
    generate_music()
