import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
# from bayes_opt import BayesianOptimization

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np

# Define the search space
search_space = [
    Real(0.0, 3.96, name='t_start'),  # 4.0 - 0.04
    Real(0.040, 4.0, name='t_end')
]

def crop_eeg(X, t_start, t_end, sfreq):
    start_idx = int(t_start * sfreq)
    end_idx = int(t_end * sfreq)
    return X[:, :, start_idx:end_idx]

def optimize_model(X_segmented, y):
    def optuna_objective(trial):
        n_components = trial.suggest_int('n_components', 2, min(X_segmented.shape[1], 8))
        
        pipeline = Pipeline([
            ('csp', CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
            ('lda', LDA())
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(pipeline, X_segmented, y, cv=cv, scoring='accuracy').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=8, show_progress_bar=True)
    return study.best_params, study.best_value

def run_bayesian_optimization(X_train, y_train, sfreq, acq_func, max_iterations=50):
    best_score = -np.inf
    best_params = {}

    @use_named_args(search_space)
    def objective(**params):
        t_start = round(params['t_start'], 3)
        t_end = round(params['t_end'], 3)

        if t_end <= t_start or (t_end - t_start) <= 0.04:
            return 1.0

        X_segmented = crop_eeg(X_train, t_start, t_end, sfreq)

        try:
            hyperparams, acc = optimize_model(X_segmented, y_train)
        except Exception:
            return 1.0

        nonlocal best_score, best_params
        if acc > best_score:
            best_score = acc
            best_params = {
                't_start': t_start,
                't_end': t_end,
                'n_components': hyperparams['n_components'],
                'accuracy': acc,
                'acq_func':acq_func
            }

        return 1.0 - acc
    result = gp_minimize(
        func=objective,
        acq_func=acq_func,
        dimensions=search_space,
        n_calls=max_iterations,
        n_initial_points=5,
        random_state=42,
        verbose=True
    )

    return best_params

def train_final_model(X_train, y_train, best_params, sfreq):
    X_train_best = crop_eeg(X_train, best_params['t_start'], best_params['t_end'], sfreq)

    model = Pipeline([
        ('csp', CSP(n_components=int(best_params['n_components']), reg=None, log=True, norm_trace=False)),
        ('svm', LDA())
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
    # plt.show()
    return kappa