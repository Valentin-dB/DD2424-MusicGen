# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:05:10 2024

@author: Valentin
"""

import os
import mido
import numpy as np

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

def files_to_songs(files,songs={}):
    for file in files:
        if file not in songs:
            try:
                midi_file = mido.MidiFile(file)
            except:
                print("Couldn't load :",file)
                continue
            matrix = []
            for msg in midi_file:
                if msg.type == "note_on":
                    matrix.append([msg.channel, msg.note, msg.velocity, msg.time])
            songs[file] = np.array(matrix)
    return songs

def dicts_from_songs(songs,velocities_and_times=False):
    Channels = set()
    Notes = set()
    if velocities_and_times:
        Velocities = set()
        Times = set()
        for song in songs.values():
            Channels.update(song[:,0])
            Notes.update(song[:,1])
            Velocities.update(song[:,2])
            Times.update(song[:,3])
    else:
        for song in songs.values():
            Channels.update(song[:,0])
            Notes.update(song[:,1])
        
    channel_to_ind = {}
    ind_to_channel = {}
    for i, channel in enumerate(sorted(Channels)):
        channel_to_ind[int(channel)] = i
        ind_to_channel[i] = int(channel)
        
    note_to_ind = {}
    ind_to_note = {}
    for i, note in enumerate(sorted(Notes)):
        note_to_ind[int(note)] = i
        ind_to_note[i] = int(note)
        
    if velocities_and_times:
        velocity_to_ind = {}
        ind_to_velocity = {}
        for i, velocity in enumerate(sorted(Velocities)):
            velocity_to_ind[int(velocity)] = i
            ind_to_velocity[i] = int(velocity)

        time_to_ind = {}
        ind_to_time = {}
        for i, time in enumerate(sorted(Times)):
            time_to_ind[time] = i
            ind_to_time[i] = time
        return channel_to_ind, ind_to_channel, note_to_ind, ind_to_note, velocity_to_ind, ind_to_velocity, time_to_ind, ind_to_time
    return channel_to_ind, ind_to_channel, note_to_ind, ind_to_note

def ranges_from_songs(songs,channels_and_notes=False):
    range_velocities = [np.inf,-np.inf]
    range_times = [np.inf,-np.inf]
    if channels_and_notes:
        range_channels = [np.inf,-np.inf]
        range_notes = [np.inf,-np.inf]
        for song in songs.values():
            min_chan = min(song[:,0])
            max_chan = max(song[:,0])
            min_note = min(song[:,1])
            max_note = max(song[:,1])
            min_vel = min(song[:,2])
            max_vel = max(song[:,2])
            min_time = min(song[:,3])
            max_time = max(song[:,3])
            if min_chan < range_channels[0] : range_channels[0] = min_chan
            if max_chan > range_channels[1] : range_channels[1] = max_chan
            if min_note < range_notes[0] : range_notes[0] = min_note
            if max_note > range_notes[1] : range_notes[1] = max_note
            if min_vel < range_velocities[0] : range_velocities[0] = min_vel
            if max_vel > range_velocities[1] : range_velocities[1] = max_vel
            if min_time < range_times[0] : range_times[0] = min_time
            if max_time > range_times[1] : range_times[1] = max_time
        return [int(a) for a in range_channels], [int(a) for a in range_notes], [int(a) for a in range_velocities], range_times
    for song in songs.values():
        min_vel = min(song[:,2])
        max_vel = max(song[:,2])
        min_time = min(song[:,3])
        max_time = max(song[:,3])
        if min_vel < range_velocities[0] : range_velocities[0] = min_vel
        if max_vel > range_velocities[1] : range_velocities[1] = max_vel
        if min_time < range_times[0] : range_times[0] = min_time
        if max_time > range_times[1] : range_times[1] = max_time
    return [int(a) for a in range_velocities], range_times

def one_hot_encode(char_to_ind,sequence):
    k = len(char_to_ind)
    if type(sequence) == int:
        one_hot = np.zeros(k)
        one_hot[char_to_ind[sequence]] = 1
        return one_hot
    else :
        n = len(sequence)
        one_hot = np.zeros((n, k))
        for i in range(n):
            one_hot[i][char_to_ind[sequence[i]]] = 1
    return one_hot

def label_sequences(one_hot,sequence_length,n_batch=1):
    n,k = one_hot.shape
    n_samples = (n-1)//sequence_length
    n_samples = (n_samples//n_batch)*n_batch
    training_set = np.reshape(one_hot[:n_samples*sequence_length,:],(n_samples,sequence_length,k))
    label_set = np.reshape(one_hot[1:n_samples*sequence_length+1,:],(n_samples,sequence_length,k))
    return training_set, label_set