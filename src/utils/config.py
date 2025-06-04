import os

# DATASET 1: BCI COMPETITION IV 2008 DATASET 2A
DATASET1_PATH=r'data\\bciiv2a_dataset\\'
DATASET1_SUBJECT = [f for f in os.listdir(DATASET1_PATH) if os.path.isfile(os.path.join(DATASET1_PATH, f))]
DATASET1_EOG = ['EOG-left', 'EOG-central', 'EOG-right']
DATASET1_CHANNEL_MAPPING = {
                            'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC4', 
                            'EEG-4': 'C5', 'EEG-5': 'F8', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 
                            'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 
                            'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz', 
                            'EEG-15': 'P2', 'EEG-16': 'POz', 'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 
                            'EOG-right': 'EOG-right'
                          }
DATASET1_STD = 'standard_1020'
DATASET1_EVENT_DICT = {
    '1023': 1,   # reject
    '1072': 2,   # eye move
    '276': 3,    # eye open
    '277': 4,    # eye close
    '32766': 5,  # new run
    '768': 6,    # new trial
    '769': 7,    # class 1 left hand
    '770': 8,    # class 2 right hand
    '771': 9,    # class 3 foot
    '772': 10,   # class 4 tongue
    '783': 11,   # unknown
}
DATASET1_USED = [i for i in range(1,9*2,2)]
DATASET1_PATH_LABEL = r'data\\bciiv2a_dataset\\true_label\\'
DATASET1_NAME = os.path.basename(os.path.dirname(DATASET1_PATH))[5:7]
DATASET1_SELECTED_CH = ['Fz', 'FC3', 'FC1', 'FCz', 'FC4', 'C5', 'F8', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

FREQ_BANDS_LIST = [[8, 13], [13, 30]]

# Bayes config
ACQ_FUNC_LIST = ["EI", "PI", "LCB"]
