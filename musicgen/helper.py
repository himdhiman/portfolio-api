from music21 import converter,instrument,note,chord,stream
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = load_model(os.path.join(BASE_DIR, "musicgen", "piano_model.hdf5"))
with open(os.path.join(BASE_DIR, "musicgen", "Notes") , 'rb') as f:
    notes = pickle.load(f)
pitchnames = sorted(set(notes))
element_to_label = dict((ele,num) for num,ele in enumerate(pitchnames))
label_to_element = dict((value,key) for (key,value) in element_to_label.items())
sequence_length = 50
network_input = []
for i in range(len(notes)-sequence_length):
    seq_in = notes[i:i+sequence_length]    
    network_input.append([element_to_label[ch] for ch in seq_in])


def resultModel(network_input, model, notes):
    Xtrain = np.reshape(network_input,(-1,len(network_input),1))
    Xtrain = Xtrain/float(len(set(notes)))
    prediction = model.predict(Xtrain)
    result = np.argmax(prediction)
    return result


def predict():
    gen_seq = []
    start = np.random.randint(len(network_input)-1)
    pattern = network_input[start]
    for i in range(100):
        ans = resultModel(pattern, model, notes)
        gen_seq.append(label_to_element[ans])
        pattern.append(ans)
        pattern = pattern[1:]
    offset = 0
    output_notes = []
    for pattern in gen_seq:
        if ('+' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('+')
            temp_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                temp_notes.append(new_note)
                
            new_chord = chord.Chord(temp_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        if offset < 7:    
            offset += 1.0
        elif offset >47:
            offset += 1.0
        else:
            offset +=0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp = os.path.join(BASE_DIR,"media","music","output.mid"))