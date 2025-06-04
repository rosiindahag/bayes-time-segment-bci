import logging
import mne

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    mne.set_log_level('WARNING')