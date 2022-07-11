import pyriemann
from mne.io import read_raw_gdf
from scipy.signal import butter, filtfilt, sosfiltfilt
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')  # for using pyplot (pip install pyqt5)


# Bandpass filtering
def butter_lowpass_filter(data, lowcut, fs, order):
    nyq = fs/2
    low = lowcut/nyq
    b, a = butter(order, low, btype='low')
    # demean before filtering
    meandat = np.mean(data, axis=1)
    data = data - meandat[:, np.newaxis]
    y = filtfilt(b, a, data) # zero-phase filter # data: [ch x time]
    return y


def butter_highpass_filter(data, highcut, fs, order):
    nyq = fs/2
    high = highcut/nyq
    b, a = butter(order, high, btype='high')
    # demean before filtering
    meandat = np.mean(data, axis=1)
    data = data - meandat[:, np.newaxis]
    y = filtfilt(b, a, data) # zero-phase filter # data: [ch x time]
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    # demean before filtering
    meandat = np.mean(data, axis=1)
    data = data - meandat[:, np.newaxis]
    y = sosfiltfilt(sos, data) # zero-phase filter # data: [ch x time]
    # specify pandlen to make the result the same as Matlab filtfilt()
    return y

# User parameters
# 769, 770, 774, 780 - left, right, up (tongue), down (feet)
#markers = [769, 770, 780, 774]
#markers_arr = {769:0, 770:1, 780:2, 774:3}
markers = [769, 770] # left, right
markers_arr = {769:1, 770:2}

# for g.tec EEG
nCh = 16
fs = 512
frame = [0.5, 3]
nTime = int((frame[1]-frame[0]) * 512)
#nTrial = 20
nClass = len(markers)
bp = [8, 30]

ch_names = ['FP1', 'FP2', 'F4', 'Fz', 'F3', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P4', 'Pz', 'P3', 'O1', 'Oz', 'O2']
roi = ['F4', 'Fz', 'F3', 'C3', 'Cz', 'C4', 'P4', 'Pz', 'P3']
roi_id = np.zeros((len(roi)), dtype=np.int64)

for i in range(len(roi)):
  roi_id[i] = ch_names.index(roi[i]) # use roi_id

nSub = 3
train_EEG = np.array([]).reshape(0, nCh, nTime)
train_labels = []

for i in range(4):
    fname = './data/s%02d/MI_run%02d.gdf' % (nSub, (i+1))
    print(fname)
    
    eeg = read_raw_gdf(fname)
    ano_types = eeg.annotations.description.astype(int) # markers
    ano_latency = np.round(eeg.annotations.onset, 4)
    eeg_times = np.round(eeg.times, 4)
    dat = eeg.get_data() * 1000000
    ch_names = eeg.ch_names

    merge_EEG = np.array([]).reshape(nCh, nTime, 0)
    for cur_markers in markers:
        event_indicies = np.argwhere(ano_types == cur_markers)
        event_latencies = ano_latency[event_indicies]
        print('current marker is '+str(cur_markers))
        n_trial = 0
        epoched_EEG = np.array([]).reshape(nCh, nTime, 0)
        tmp_labels = markers_arr[cur_markers] * np.ones((len(event_latencies)))
        train_labels = np.append(train_labels, tmp_labels)  

        for cur_latency in event_latencies:
            m_onset = np.where(eeg_times == cur_latency)[0][0]
            tmp_epoch = dat[:, m_onset + int(frame[0]*fs):m_onset + int(frame[1]*fs)]

            # epoch-level bandpass filtering
            tmp_epoch = butter_bandpass_filter(tmp_epoch, bp[0], bp[1], fs, 4)
            epoched_EEG = np.dstack((epoched_EEG, tmp_epoch))
            n_trial = n_trial + 1
        merge_EEG = np.dstack((merge_EEG, epoched_EEG))
    
    merge_EEG = np.transpose(merge_EEG, (2, 0, 1)) # now [trial x ch x time]
    train_EEG = np.vstack((train_EEG, merge_EEG))

cov_train = pyriemann.estimation.Covariances().fit_transform(train_EEG[:, roi_id, :])
print(cov_train.shape)
print(train_labels.shape)

mdm = pyriemann.classification.MDM()
mdm.metric = 'Riemann'
mdm.fit(cov_train, train_labels) # training

mdm_train_acc = np.sum(mdm.predict(cov_train) == train_labels) / len(train_labels) # train - meaningless
print('training accuracy is', np.round(mdm_train_acc,4))

trained = {'COV':cov_train, 'Labels':train_labels}
fname_user = input('Enter model name: ')
fname_model = './data/s%02d/%s.pkl' % (nSub, fname_user)
print(fname_model, 'saved.')
out_file = open(fname_model, 'wb')
pickle.dump(trained, out_file)
out_file.close()

