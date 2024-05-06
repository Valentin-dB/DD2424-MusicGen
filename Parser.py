# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:05:10 2024

@author: Valentin
"""

import os
import mido

def get_midi_files(files):
    MIDI_EXTENSIONS = [".mid", ".midi", ".MID", ".MIDI"]
    midi_files = []
    for file_name in files:
        # Check if it is a midi file
        _, extension = os.path.splitext(file_name)
        if extension in MIDI_EXTENSIONS:
            midi_files.append(file_name)
    return midi_files

def load_dataset(path):
    dataset = {}

    # Read all MIDI songs
    for dir, _ , files in os.walk(path):
        if len(files) == 0:
            continue
        
        genre    = dir.split("\\")[-3]

        # Add full path to all midi files
        full_path_files = []
        for filename in get_midi_files(files):
            full_path_files.append(os.path.join(dir, filename))

        if genre not in dataset:
            dataset[genre] = full_path_files
        else:
            dataset[genre] += full_path_files

    return dataset

def files_to_songs(files):
    songs = {}
    for file in files:
        midi_file = mido.MidiFile(file)
        ticks_per_beat = midi_file.ticks_per_beat
        matrix = []
        for msg in midi_file:
            if msg.type == "note_on":
                matrix.append([msg.channel, msg.note, msg.velocity, int(msg.time)*ticks_per_beat])
        songs[file] = matrix
    return songs

dataset = load_dataset("adl-piano-midi")
songs = files_to_songs(dataset["Reggae"])