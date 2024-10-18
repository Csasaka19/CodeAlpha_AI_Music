from music21 import converter, note, chord
import numpy as np
import glob
import pickle

def get_notes():
    """Extract notes and chords from MIDI files in the data/midi_files/ directory."""
    notes = []
    midi_files = glob.glob("data/midi_files/*.mid")  # Path updated according to structure

    for file in midi_files:
        midi = converter.parse(file)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)
        if parts:  # If the file has instrument parts, extract notes
            notes_to_parse = parts.parts[0].recurse()
        else:  # File has no instrument parts
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # Convert chord to string of notes
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # Save notes to a file
    with open('data/processed/notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, sequence_length=100):
    """Prepare sequences from the extracted notes for training the RNN."""
    pitch_names = sorted(set(note for note in notes))
    n_vocab = len(pitch_names)

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    input_sequences = []
    output_notes = []

    # Create input-output pairs from sequences of notes
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input_sequences.append([note_to_int[char] for char in sequence_in])
        output_notes.append(note_to_int[sequence_out])

    # Reshape the input into format acceptable by LSTM
    n_patterns = len(input_sequences)
    input_sequences = np.reshape(input_sequences, (n_patterns, sequence_length, 1))
    input_sequences = input_sequences / float(n_vocab)  # Normalize the input

    output_notes = np_utils.to_categorical(output_notes)

    # Save processed data to files
    np.save('data/processed/input_sequences.npy', input_sequences)
    np.save('data/processed/output_notes.npy', output_notes)

    return input_sequences, output_notes, n_vocab
