import numpy as np
import PreProcessor as pp
import tensorflow as tf
import matplotlib.pyplot as plt

def sample_probas(vec, mode, T):
    output = []
    if mode == "Max":
        channels = np.zeros_like(vec[0])
        i = np.argmax(vec[0][0,0,:])
        channels[0,0,i] = 1
        output += [channels]
        
        notes = np.zeros_like(vec[1])
        i = np.argmax(vec[1][0,0,:])
        notes[0,0,i] = 1
        output += [notes]
    
    elif mode == "Original":
        p = vec[0][0,0,:]
        channels = np.zeros_like(vec[0])
        i = np.random.choice(range(len(p)),p=p)
        channels[0,0,i] = 1
        output += [channels]
        
        p = vec[1][0,0,:]
        notes = np.zeros_like(vec[1])
        i = np.random.choice(range(len(p)),p=p)
        notes[0,0,i] = 1
        output += [notes]
    
    elif mode == "SoftMax":
        q = np.log(vec[0][0,0,:])
        p = np.exp((q - np.max(q))/T)
        p = p/np.sum(p)
        channels = np.zeros_like(vec[0])
        i = np.random.choice(range(len(p)),p=p)
        channels[0,0,i] = 1
        output += [channels]
        
        q = np.log(vec[1][0,0,:])
        p = np.exp((q - np.max(q))/T)
        p = p/np.sum(p)
        notes = np.zeros_like(vec[1])
        i = np.random.choice(range(len(p)),p=p)
        notes[0,0,i] = 1
        output += [notes]
        
    elif mode == "NucleusSampling":
        p = vec[0][0,0,:]
        sorted_p = np.sort(p)[::-1]
        cum_p = np.cumsum(sorted_p)
        k = np.sum(cum_p < T) + 1
        indices = np.argsort(p)[:-k]
        p[indices] = 0
        i = np.random.choice(range(len(p)),p=p/cum_p[k-1])
        channels = np.zeros_like(vec[0])
        channels[0,0,i] = 1
        output += [channels]
        
        p = vec[1][0,0,:]
        sorted_p = np.sort(p)[::-1]
        cum_p = np.cumsum(sorted_p)
        k = np.sum(cum_p < T) + 1
        indices = np.argsort(p)[:-k]
        p[indices] = 0
        i = np.random.choice(range(len(p)),p=p/cum_p[k-1])
        notes = np.zeros_like(vec[1])
        notes[0,0,i] = 1
        output += [notes]
        
    output += [np.rint(vec[2]).astype(int)]
    output += [vec[3]]
    return output

def generate(model,input_vec,output_length,ind_to_channel,ind_to_note,mode="Max",T=1,one_hot=False,reset=False):
    stateful = model.stateful
    model.stateful = True
    if reset : model.reset_states()
    temp_vec = model.predict(input_vec)
    for i in range(4):
        temp_vec[i] = temp_vec[i][:,-1:,:]
    temp_vec = sample_probas(temp_vec,mode,T)
    if one_hot :
        output_vec = temp_vec
    else :
        output_vec = [[ind_to_channel[np.argmax(temp_vec[0][0,0,:])],ind_to_note[np.argmax(temp_vec[1][0,0,:])],temp_vec[2][0,0,0],temp_vec[3][0,0,0]]]
    for i in range(output_length-1):
        temp_vec = model.predict(temp_vec)
        temp_vec = sample_probas(temp_vec,mode,T)
        if one_hot :
            for j in range(4):
                output_vec[j] = np.concatenate((output_vec[j],temp_vec[j]),axis=1)
        else :
            output_vec += [[ind_to_channel[np.argmax(temp_vec[0][0,0,:])],ind_to_note[np.argmax(temp_vec[1][0,0,:])],temp_vec[2][0,0,0],temp_vec[3][0,0,0]]]
    model.stateful = stateful
    return output_vec
    
def create_midi(gen):
    channels = []
    notes = []
    velocities = []
    times = []
    for i in range(len(gen)):
        channels += [gen[i][0]]
        notes += [gen[i][1]]
        velocities += [gen[i][2]]
        times += [gen[i][3]]
    return channels, notes, velocities, times

def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    correct_predictions = tf.equal(tf.cast(y_pred_rounded, tf.int32), tf.cast(y_true, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def tol_accuracy(y_true, y_pred):
    threshold = 0.1
    difference = tf.abs(tf.subtract(y_true, y_pred)) - threshold
    correct_predictions = tf.where(difference <= 0, True, False)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def plot_training_results(title,loss,channels_accuracy,notes_accuracy,velocities_accuracy,times_accuracy,val_loss,val_channels_accuracy,val_notes_accuracy,val_velocities_accuracy,val_times_accuracy):
    n = len(loss)+1
    x = range(1,n)
    x_ticks = range(1,n,max(1,n//10))

    plt.figure()
    plt.plot(x, loss, label="Training loss")
    plt.plot(x, val_loss, label="Validation loss")
    plt.title(title)
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.xticks(x_ticks)
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(9, 8))

    axs[0,0].plot(x, channels_accuracy, label="Training set")
    axs[0,0].plot(x, val_channels_accuracy, label="Validation set")
    axs[0,0].legend()
    axs[0,0].set_ylim([0,1])
    axs[0,0].set_title("Channels")
    axs[0,0].set_xlabel("Number of epochs")
    axs[0,0].set_ylabel("Accuracy")
    axs[0,0].set_xticks(x_ticks)

    axs[0,1].plot(x, notes_accuracy, label="Training set")
    axs[0,1].plot(x, val_notes_accuracy, label="Validation set")
    axs[0,1].legend()
    axs[0,1].set_ylim([0,1])
    axs[0,1].set_title("Notes")
    axs[0,1].set_xlabel("Number of epochs")
    axs[0,1].set_ylabel("Accuracy")
    axs[0,1].set_xticks(x_ticks)

    axs[1,0].plot(x, velocities_accuracy, label="Training set")
    axs[1,0].plot(x, val_velocities_accuracy, label="Validation set")
    axs[1,0].legend()
    axs[1,0].set_ylim([0,1])
    axs[1,0].set_title("Velocity")
    axs[1,0].set_xlabel("Number of epochs")
    axs[1,0].set_ylabel("Accuracy")
    axs[1,0].set_xticks(x_ticks)

    axs[1,1].plot(x, times_accuracy, label="Training set")
    axs[1,1].plot(x, val_times_accuracy, label="Validation set")
    axs[1,1].legend()
    axs[1,1].set_ylim([0,1])
    axs[1,1].set_title("Time")
    axs[1,1].set_xlabel("Number of epochs")
    axs[1,1].set_ylabel("Accuracy")
    axs[1,1].set_xticks(x_ticks)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 1])


def prep_data(seq_length, ClassicSongs,n_Channels,n_Notes,channel_to_ind,note_to_ind,val_split=0.1,test_split=0.1):
    total_samples = sum((len(song) - 1) // seq_length for song in ClassicSongs.values())

    X_Channels = np.zeros((total_samples, seq_length, n_Channels))
    X_Notes = np.zeros((total_samples, seq_length, n_Notes))
    X_Velocities = np.zeros((total_samples, seq_length, 1))
    X_Times = np.zeros((total_samples, seq_length, 1))
    y_Channels = np.zeros((total_samples, seq_length, n_Channels))
    y_Notes = np.zeros((total_samples, seq_length, n_Notes))
    y_Velocities = np.zeros((total_samples, seq_length, 1))
    y_Times = np.zeros((total_samples, seq_length, 1))

    current_index = 0
    for song in ClassicSongs.values():
        song_x_channels, song_y_channels = pp.label_sequences(pp.one_hot_encode(channel_to_ind, song[:, 0]), seq_length)
        song_x_notes, song_y_notes = pp.label_sequences(pp.one_hot_encode(note_to_ind, song[:, 1]), seq_length)
        song_x_velocities, song_y_velocities = pp.label_sequences(song[:, 2:3], seq_length)
        song_x_ticks, song_y_ticks = pp.label_sequences(song[:, 3:], seq_length)
        
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