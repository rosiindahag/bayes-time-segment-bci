from src.models.optimization import run_bayesian_optimization
from src.models.csp_svm import train_final_model, evaluate_model
from src.utils.config import *
from src.utils.logger import setup_logging
from src.data_preprocessing.preprocessing import load_raw_file, filter_data, get_epoch, get_test_label

from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

def main():
    setup_logging()
    
    # Subject setup
    subject_id = 1
    file_name_train = DATASET1_SUBJECT[subject_id]
    file_name_test = DATASET1_SUBJECT[subject_id - 1]
    subject = file_name_train.split(".")[0][:3]

    acq_func = ACQ_FUNC_LIST[0]
    min_freq, max_freq = FREQ_BANDS_LIST[0]
    print(f"SUBJECT: {subject}")

    # Load data
    raw_train, events_train, event_id_train = load_raw_file(DATASET1_PATH, 
                                                        file_name_train, 
                                                        DATASET1_CHANNEL_MAPPING,
                                                        DATASET1_STD, 
                                                        DATASET1_SELECTED_CH, 
                                                        DATASET1_EVENT_DICT, 
                                                        DATASET1_EOG)
    raw_test, events_test, event_id_test = load_raw_file(DATASET1_PATH, 
                                                     file_name_test, 
                                                     DATASET1_CHANNEL_MAPPING, 
                                                     DATASET1_STD, 
                                                     DATASET1_SELECTED_CH, 
                                                     DATASET1_EVENT_DICT, 
                                                     DATASET1_EOG)
    sfreq = raw_train.info['sfreq']

    print(f"Processing frequency band {min_freq}-{max_freq} Hz")
    # Filter
    raw_train_f = filter_data(raw_train, min_freq, max_freq)
    raw_test_f = filter_data(raw_test, min_freq, max_freq)

    # Epoching
    epochs_train = get_epoch(raw_train_f, events_train, [7, 8, 9, 10])
    epochs_test = get_epoch(raw_test_f, events_test, [11])

    # Labeling
    le = LabelEncoder()
    y_train = le.fit_transform(epochs_train.events[:, -1])
    X_train = epochs_train.get_data(copy=False)

    temp_X_test = epochs_test.get_data(copy=False)

    y_test_dict = get_test_label(DATASET1_PATH_LABEL, f'{subject}E', 288, 1, 2, 3, 4) #288 trials, class1=left, class2=right
    y_test = [label for label in y_test_dict.values()]

    y_test = le.fit_transform(y_test)
    y_test_idx = [idx for idx in y_test_dict.keys()]

    X_test = [temp_X_test[i] for i in y_test_idx]
    X_test = np.array(X_test)

    folder_path = f"results\\bayescspsvm\\{subject}\\{acq_func}_{min_freq}_{max_freq}\\"

    # Optimize and train
    best_params = run_bayesian_optimization(X_train, y_train, sfreq)
    model = train_final_model(X_train, y_train, best_params, sfreq)
    evaluate_model(model, X_test, y_test, best_params, sfreq, subject, folder_path)

    with open(rf"{folder_path}\\best_eeg_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()
