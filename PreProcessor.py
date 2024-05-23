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
                #print("Couldn't load :",file)
                continue
            matrix = []
            for msg in midi_file:
                if msg.type == "note_on":
                    matrix.append([msg.channel, msg.note, msg.velocity, msg.time])
            songs[file] = np.array(matrix)
    return songs

def dicts_from_songs(songs,velocities=True,times=False):
    Channels = set()
    Notes = set()
    if velocities : Velocities = set()
    if times : Times = set()
    for song in songs.values():
        Channels.update(song[:,0])
        Notes.update(song[:,1])
        if velocities : Velocities.update(song[:,2])
        if times : Times.update(song[:,3])
        
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
        
    if velocities :
        velocity_to_ind = {}
        ind_to_velocity = {}
        for i, velocity in enumerate(sorted(Velocities)):
            velocity_to_ind[int(velocity)] = i
            ind_to_velocity[i] = int(velocity)
    if times : 
        time_to_ind = {}
        ind_to_time = {}
        for i, time in enumerate(sorted(Times)):
            time_to_ind[time] = i
            ind_to_time[i] = time
    if velocities and times : return channel_to_ind, ind_to_channel, note_to_ind, ind_to_note, velocity_to_ind, ind_to_velocity, time_to_ind, ind_to_time
    if velocities : return channel_to_ind, ind_to_channel, note_to_ind, ind_to_note, velocity_to_ind, ind_to_velocity
    if times : return channel_to_ind, ind_to_channel, note_to_ind, ind_to_note, time_to_ind, ind_to_time
    return channel_to_ind, ind_to_channel, note_to_ind, ind_to_note

def ranges_from_songs(songs,channels=False,notes=False,velocities=False,times=True):
    res = ()
    n_true = channels+notes+velocities+times
    if channels :
        range_channels = [np.inf,-np.inf]
        for song in songs.values():
            min_chan = min(song[:,0])
            max_chan = max(song[:,0])
            if min_chan < range_channels[0] : range_channels[0] = min_chan
            if max_chan > range_channels[1] : range_channels[1] = max_chan
        if n_true == 1 : return [int(a) for a in range_channels]
        res += ([int(a) for a in range_channels],)
    if notes :
        range_notes = [np.inf,-np.inf]
        for song in songs.values():
            min_note = min(song[:,1])
            max_note = max(song[:,1])
            if min_note < range_notes[0] : range_notes[0] = min_note
            if max_note > range_notes[1] : range_notes[1] = max_note
        if n_true == 1 : return [int(a) for a in range_notes]
        res += ([int(a) for a in range_notes],)
    if velocities :
        range_velocities = [np.inf,-np.inf]
        for song in songs.values():
            min_vel = min(song[:,2])
            max_vel = max(song[:,2])
            if min_vel < range_velocities[0] : range_velocities[0] = min_vel
            if max_vel > range_velocities[1] : range_velocities[1] = max_vel
        if n_true == 1 : return [int(a) for a in range_velocities]
        res += ([int(a) for a in range_velocities],)
    if times :
        range_times = [np.inf,-np.inf]
        for song in songs.values():
            min_time = min(song[:,3])
            max_time = max(song[:,3])
            if min_time < range_times[0] : range_times[0] = min_time
            if max_time > range_times[1] : range_times[1] = max_time
        if n_true == 1 : return range_times
        res += (range_times,)
    return res

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

def one_hot_encode_no_dict(sequence,k):
    if type(sequence) == int:
        one_hot = np.zeros(k)
        one_hot[sequence] = 1
        return one_hot
    else :
        n = len(sequence)
        one_hot = np.zeros((n, k))
        for i in range(n):
            one_hot[i][sequence[i]] = 1
    return one_hot

def label_sequences(one_hot,sequence_length,n_batch=1):
    n,k = one_hot.shape
    n_samples = (n-1)//sequence_length
    n_samples = (n_samples//n_batch)*n_batch
    training_set = np.reshape(one_hot[:n_samples*sequence_length,:],(n_samples,sequence_length,k))
    label_set = np.reshape(one_hot[1:n_samples*sequence_length+1,:],(n_samples,sequence_length,k))
    return training_set, label_set

def label_sequences_transformer(x,sequence_length,to_one_hot=None,n_batch=1):
    n = x.shape[0]
    n_samples = (n-1)//sequence_length
    n_samples = (n_samples//n_batch)*n_batch
    training_set = np.reshape(x[:n_samples*sequence_length],(n_samples,sequence_length))
    if to_one_hot is not None :
        x = one_hot_encode_no_dict(x,to_one_hot)
        label_set = np.reshape(x[1:n_samples*sequence_length+1,:],(n_samples,sequence_length,-1))
    else :
        label_set = np.reshape(x[1:n_samples*sequence_length+1],(n_samples,sequence_length,1))
    return training_set, label_set

def prep_data(seq_length,ClassicSongs,n_Channels,n_Notes,n_Velocities,channel_to_ind,note_to_ind,velocity_to_ind,val_split=0.1,test_split=0.1):
    if type(ClassicSongs) == list : ClassicSongs = files_to_songs(ClassicSongs)
        
    total_samples = sum((len(song) - 1) // seq_length for song in ClassicSongs.values())

    X_Channels = np.zeros((total_samples, seq_length, n_Channels))
    X_Notes = np.zeros((total_samples, seq_length, n_Notes))
    X_Velocities = np.zeros((total_samples, seq_length, n_Velocities))
    X_Times = np.zeros((total_samples, seq_length, 1))
    y_Channels = np.zeros((total_samples, seq_length, n_Channels))
    y_Notes = np.zeros((total_samples, seq_length, n_Notes))
    y_Velocities = np.zeros((total_samples, seq_length, n_Velocities))
    y_Times = np.zeros((total_samples, seq_length, 1))

    current_index = 0
    for song in ClassicSongs.values():
        song_x_channels, song_y_channels = label_sequences(one_hot_encode(channel_to_ind, song[:, 0]), seq_length)
        song_x_notes, song_y_notes = label_sequences(one_hot_encode(note_to_ind, song[:, 1]), seq_length)
        song_x_velocities, song_y_velocities = label_sequences(one_hot_encode(velocity_to_ind, song[:, 2]), seq_length)
        song_x_ticks, song_y_ticks = label_sequences(song[:, 3:], seq_length)
        
        n_samples = song_x_channels.shape[0]
        next_index = current_index + n_samples
        
        X_Channels[current_index:next_index] = song_x_channels
        X_Notes[current_index:next_index] = song_x_notes
        X_Velocities[current_index:next_index] = song_x_velocities
        X_Times[current_index:next_index] = song_x_ticks
        y_Channels[current_index:next_index] = song_y_channels
        y_Notes[current_index:next_index] = song_y_notes
        y_Velocities[current_index:next_index] = song_y_velocities
        y_Times[current_index:next_index] = song_y_ticks
        
        current_index = next_index

    train_split = 1 - val_split - test_split

    n_val = int(total_samples*val_split)
    n_test = int(total_samples*test_split)
    indices = np.random.permutation(total_samples)
    val_indices = indices[:n_val]
    test_indices = indices[-n_test:]
    train_indices = indices[n_val:-n_test]

    Test_X_Channels = X_Channels[test_indices,:,:]
    Test_X_Notes = X_Notes[test_indices,:,:]
    Test_X_Velocities = X_Velocities[test_indices,:,:]
    Test_X_Times = X_Times[test_indices,:,:]
    Test_y_Channels = y_Channels[test_indices,:,:]
    Test_y_Notes = y_Notes[test_indices,:,:]
    Test_y_Velocities = y_Velocities[test_indices,:,:]
    Test_y_Times = y_Times[test_indices,:,:]

    Val_X_Channels = X_Channels[val_indices,:,:]
    Val_X_Notes = X_Notes[val_indices,:,:]
    Val_X_Velocities = X_Velocities[val_indices,:,:]
    Val_X_Times = X_Times[val_indices,:,:]
    Val_y_Channels = y_Channels[val_indices,:,:]
    Val_y_Notes = y_Notes[val_indices,:,:]
    Val_y_Velocities = y_Velocities[val_indices,:,:]
    Val_y_Times = y_Times[val_indices,:,:]


    X_Channels = X_Channels[train_indices,:,:]
    X_Notes = X_Notes[train_indices,:,:]
    X_Velocities = X_Velocities[train_indices,:,:]
    X_Times = X_Times[train_indices,:,:]
    y_Channels = y_Channels[train_indices,:,:]
    y_Notes = y_Notes[train_indices,:,:]
    y_Velocities = y_Velocities[train_indices,:,:]
    y_Times = y_Times[train_indices,:,:]

    return X_Channels, X_Notes, X_Velocities, X_Times, y_Channels, y_Notes, y_Velocities, y_Times, Val_X_Channels, Val_X_Notes, Val_X_Velocities, Val_X_Times, Val_y_Channels, Val_y_Notes, Val_y_Velocities, Val_y_Times, Test_X_Channels, Test_X_Notes, Test_X_Velocities, Test_X_Times, Test_y_Channels, Test_y_Notes, Test_y_Velocities, Test_y_Times

def prep_data_transformer(seq_length,ClassicSongs,n_Channels,n_Notes,n_Velocities=None,val_split=0.1,test_split=0.1):
    if type(ClassicSongs) == list : ClassicSongs = files_to_songs(ClassicSongs)
        
    total_samples = sum((len(song) - 1) // seq_length for song in ClassicSongs.values())

    X_Channels = np.zeros((total_samples, seq_length))
    X_Notes = np.zeros((total_samples, seq_length))
    X_Velocities = np.zeros((total_samples, seq_length))
    X_Times = np.zeros((total_samples, seq_length))
    y_Channels = np.zeros((total_samples, seq_length, n_Channels))
    y_Notes = np.zeros((total_samples, seq_length, n_Notes))
    if n_Velocities is not None : y_Velocities = np.zeros((total_samples, seq_length, n_Velocities))
    else : y_Velocities = np.zeros((total_samples, seq_length, 1))
    y_Times = np.zeros((total_samples, seq_length, 1))

    current_index = 0
    for song in ClassicSongs.values():
        if song.shape[0] <= seq_length : continue
        song_x_channels, song_y_channels = label_sequences_transformer(song[:, 0].astype(int), seq_length, n_Channels)
        song_x_notes, song_y_notes = label_sequences_transformer(song[:, 1].astype(int), seq_length, n_Notes)
        song_x_velocities, song_y_velocities = label_sequences_transformer(song[:, 2].astype(int), seq_length, n_Velocities)
        song_x_ticks, song_y_ticks = label_sequences_transformer(song[:, 3], seq_length)
        
        n_samples = song_x_channels.shape[0]
        next_index = current_index + n_samples
        
        X_Channels[current_index:next_index] = song_x_channels
        X_Notes[current_index:next_index] = song_x_notes
        X_Velocities[current_index:next_index] = song_x_velocities
        X_Times[current_index:next_index] = song_x_ticks
        y_Channels[current_index:next_index] = song_y_channels
        y_Notes[current_index:next_index] = song_y_notes
        y_Velocities[current_index:next_index] = song_y_velocities
        y_Times[current_index:next_index] = song_y_ticks
        
        current_index = next_index

    train_split = 1 - val_split - test_split

    n_val = int(total_samples*val_split)
    n_test = int(total_samples*test_split)
    indices = np.random.permutation(total_samples)
    val_indices = indices[:n_val]
    test_indices = indices[-n_test:]
    train_indices = indices[n_val:-n_test]

    Test_X = [X_Channels[test_indices,:],X_Notes[test_indices,:],X_Velocities[test_indices,:],X_Times[test_indices,:]]
    Test_y = [y_Channels[test_indices,:,:],y_Notes[test_indices,:,:],y_Velocities[test_indices,:],y_Times[test_indices,:]]

    Val_X = [X_Channels[val_indices,:],X_Notes[val_indices,:],X_Velocities[val_indices,:],X_Times[val_indices,:]]
    Val_y = [y_Channels[val_indices,:,:],y_Notes[val_indices,:,:],y_Velocities[val_indices,:],y_Times[val_indices,:]]

    Train_X = [X_Channels[train_indices,:],X_Notes[train_indices,:],X_Velocities[train_indices,:],X_Times[train_indices,:]]
    Train_y = [y_Channels[train_indices,:,:],y_Notes[train_indices,:,:],y_Velocities[train_indices,:],y_Times[train_indices,:]]

    return Train_X, Train_y, Val_X, Val_y, Test_X, Test_y
