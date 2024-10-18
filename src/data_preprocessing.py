import numpy as np
from music21 import converter, instrument, note, chord
import glob
import pickle

midi_path = "data/midi_files/*.mid"

notes = []


for midi_file in glob.glob(midi_path):
    midi = converter.parse(midi_file)
    notes_to_parse = None

    parts = instrument.partitionByInstrument(midi)

    if parts:  # If there are instrument parts
        notes_to_parse = parts.parts[0].recurse() 
    else:
        notes_to_parse = midi.flat.notes 

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder)) 


with open("data/processed/notes.pkl", "wb") as filepath:
    pickle.dump(notes, filepath)

print("Notes extracted and saved to 'data/processed/notes.pkl'")
