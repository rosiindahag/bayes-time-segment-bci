import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP
from sklearn.svm import SVC
# from bayes_opt import BayesianOptimization

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np

# Define the search space
search_space = [
    Real(0.0, 3.96, name='t_start'),  # 4.0 - 0.04
    Real(1.0, 4.0, name='t_end')
]

best_score = -np.inf
best_params = {}

def crop_eeg(X, t_start, t_end, sfreq):
    start_idx = int(t_start * sfreq)
    end_idx = int(t_end * sfreq)
    return X[:, :, start_idx:end_idx]

def optimize_model(X_segmented, y):
    def optuna_objective(trial):
        n_components = trial.suggest_int('n_components', 2, min(X_segmented.shape[1], 8))
        C = trial.suggest_float('C', 1e-1, 1e3, log=True)
        kernel = trial.suggest_categorical('kernel', ["linear", "rbf", "poly"])
        gamma = trial.suggest_float('gamma', 1e-4, 1e-1, log=True) if kernel in ["rbf", "poly"] else "scale"

        pipeline = Pipeline([
            ('csp', CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
            ('svm', SVC(kernel=kernel, C=C, gamma=gamma))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(pipeline, X_segmented, y, cv=cv, scoring='accuracy').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=30, show_progress_bar=True)
    return study.best_params, study.best_value

def run_bayesian_optimization(X_train, y_train, sfreq, acq_func, max_iterations=50):
    global best_score, best_params

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

        global best_score, best_params
        if acc > best_score:
            best_score = acc
            best_params = {
                't_start': t_start,
                't_end': t_end,
                'n_components': hyperparams['n_components'],
                'C': hyperparams['C'],
                'kernel': hyperparams['kernel'],
                'accuracy': acc,
                'gamma': hyperparams.get('gamma', 'scale'),
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

# def run_bayesian_optimization(X_train, y_train, sfreq, max_iterations=50):
#     def objective(t_start, t_end):
#         global best_score, best_params
#         if t_end <= t_start:
#             return -1.0
#         if (t_end - t_start) <= 0.04:
#             return -1.0
#         X_segmented = crop_eeg(X_train, t_start, t_end, sfreq)
#         try:
#             hyperparams, acc = optimize_model(X_segmented, y_train)
#         except Exception:
#             return -1.0

#         if acc > best_score:
#             best_score = acc
#             best_params = {
#                 't_start': t_start,
#                 't_end': t_end,
#                 'n_components': hyperparams['n_components'],
#                 'C': hyperparams['C'],
#                 'kernel': hyperparams['kernel'],
#                 'accuracy': acc
#             }
#             best_params['gamma'] = hyperparams.get('gamma', 'scale')
#         return acc

#     optimizer = BayesianOptimization(
#         f=objective,
#         pbounds={'t_start': (0.0, 4.0 - 0.04), 't_end': (1.0, 4.0)},
#         verbose=2,
#         random_state=42
#     )
#     optimizer.maximize(init_points=5, n_iter=max_iterations)
#     return best_params