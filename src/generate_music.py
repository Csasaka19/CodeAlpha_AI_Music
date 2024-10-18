import numpy as np
from keras.models import load_model
from music21 import instrument, note, stream, chord
import pickle
import random

# Load the preprocessed notes
with open("data/processed/notes.pkl", "rb") as filepath:
    notes = pickle.load(filepath)

# Load the trained model
model = load_model("model/music_generator_model.h5")

# Get all the pitch names and unique note count
n_vocab = len(set(notes))
note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}
int_to_note = {number: note for note, number in note_to_int.items()}

# Seed for generating new music (random sequence from the data)
start = random.randint(0, len(notes) - 100)
pattern = [note_to_int[note] for note in notes[start:start + 100]]

# Generate new notes
generated_notes = []

for i in range(500):  # Generate 500 notes
    input_sequence = np.reshape(pattern, (1, len(pattern), 1))
    input_sequence = input_sequence / float(n_vocab)

    prediction = model.predict(input_sequence, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]

    generated_notes.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# Convert the output to MIDI file
offset = 0
output_notes = []

for pattern in generated_notes:
    if ('.' in pattern) or pattern.isdigit():
        chords = pattern.split('.')
        notes = [note.Note(int(chord)) for chord in chords]
        for note_obj in notes:
            note_obj.storedInstrument = instrument.Piano()
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5  # Increase the offset for the next note

# Create a stream object
midi_stream = stream.Stream(output_notes)

# Write the MIDI file to the output directory
midi_stream.write('midi', fp='output/generated_music.mid')
print("Generated music saved to 'output/generated_music.mid'")
