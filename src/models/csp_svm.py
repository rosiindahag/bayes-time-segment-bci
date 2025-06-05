import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from mne.decoding import CSP
from .optimization import crop_eeg

def train_final_model(X_train, y_train, best_params, sfreq):
    X_train_best = crop_eeg(X_train, best_params['t_start'], best_params['t_end'], sfreq)

    model = Pipeline([
        ('csp', CSP(n_components=int(best_params['n_components']), reg=None, log=True, norm_trace=False)),
        ('svm', SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma']))
    ])
    model.fit(X_train_best, y_train)
    return model

def evaluate_model(model, X_test, y_test, best_params, sfreq, subject, folder_path):
    # from .optimization import crop_eeg
    X_test_best = crop_eeg(X_test, best_params['t_start'], best_params['t_end'], sfreq)

    y_pred = model.predict(X_test_best)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Final Kappa Score on Test Set: {kappa:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left', 'Right', 'Foot', 'Tongue'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Subject {subject}")

    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "confusion_matrix.png"))
    plt.show()
    return kappa