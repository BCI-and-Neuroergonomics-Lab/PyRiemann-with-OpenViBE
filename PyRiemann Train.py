# generic import
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# mne import
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io import read_raw_gdf

# pyriemann import
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

# Establish number of classes (2, 3, or 4)
desired = ["769", "770", "780"]  # , "774"]
#           left, right, up,          down


# Function for pre-processing EEG data and extracting epochs
def pymann(raw, desired, tmin, tmax):
    # set to standard 10 20 montage in case we need it
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Apply band-pass filter
    raw.filter(7., 35., method='iir', picks=picks)

    events, temp_id = events_from_annotations(raw)
    event_id = [temp_id.get(i) for i in desired]

    # Read epochs
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False)
    labels = epochs.events[:, -1]

    # scale epochs up for classification
    epochs_data_train = 1e6 * epochs.get_data()

    # compute covariance matrices
    return Covariances().transform(epochs_data_train), labels


########################################
# Train Riemannian Geometry Classifier #
########################################

# avoid classification of evoked responses by using epochs that start 1s after cue onset
tmin, tmax = 1.0, 3.5

raw_files = [
    read_raw_gdf("./Data/sub02/sess02/MI_run01.gdf", preload=True),
    read_raw_gdf("./Data/sub02/sess02/MI_run02.gdf", preload=True),
    read_raw_gdf("./Data/sub02/sess02/MI_run03.gdf", preload=True),
    read_raw_gdf("./Data/sub02/sess02/MI_run04.gdf", preload=True),
    read_raw_gdf("./Data/sub02/sess02/MI_run05.gdf", preload=True)
]

raw = concatenate_raws(raw_files)

# Train MDM classifier on covariance matrix
cov_data_train, train_labels = pymann(raw, desired, tmin, tmax)
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))  # Minimum Distance to Mean classifier
mdm.fit(cov_data_train, train_labels)

# Now load a new file for the test accuracy
test_file = read_raw_gdf("./Data/sub02/sess02/MI_run06.gdf", preload=True)

# Repeat from before, but with test file
cov_data_test, test_labels = pymann(test_file, desired, tmin, tmax)


#####################
# Report Accuracies #
#####################

# Training accuracy, always overestimates...
mdm_train_acc = np.sum(mdm.predict(cov_data_train) == train_labels) / len(train_labels)
print('training accuracy is', np.round(mdm_train_acc, 4))

# Test set accuracy, represents completely new data
mdm_acc = np.sum(mdm.predict(cov_data_test) == test_labels) / len(test_labels)
print('test accuracy is', np.round(mdm_acc, 4))

# Show class #1 correlation matrix, repeat to show others (change [0] to [1], etc)...
#df = pd.DataFrame(data=mdm.covmeans_[0], index=raw.ch_names, columns=raw.ch_names)
#plt.matshow(df.corr())
#plt.show()

###################
# Save Classifier #
###################

#classifier = {'COV': cov_data_train, 'Labels': train_labels}
#fname = input('Enter file name (include subject ID): ')
#path = "./Data/sub02/" + fname + ".pkl"
#out_file = open(path, 'wb')
#pickle.dump(classifier, out_file)
#out_file.close()
