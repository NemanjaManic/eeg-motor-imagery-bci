"""Configuration constants for the EEG motor imagery BCI application.

Centralizing these here means switching hardware (COM port) or tuning the
experiment protocol never requires touching the application logic.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

TRAINING_DATA_PATH = DATA_DIR / "eeg_training_data.csv"
TEST_INSTRUCTIONS_PATH = DATA_DIR / "test_instructions.csv"

LEFT_NEUTRAL_IMAGE = ASSETS_DIR / "left_neutral.png"
LEFT_ACTIVE_IMAGE = ASSETS_DIR / "left_active.png"
RIGHT_NEUTRAL_IMAGE = ASSETS_DIR / "right_neutral.png"
RIGHT_ACTIVE_IMAGE = ASSETS_DIR / "right_active.png"

# --- SMARTING serial connection ---
COM_PORT = "COM4"
BAUD_RATE = 921600
SAMPLING_RATE_HZ = 160
PACKET_SIZE = 83

# --- ADC-to-microvolts conversion (SMARTING protocol) ---
VOLTAGE_REFERENCE = 4.5
GAIN = 24

# --- Channel byte offsets within a data packet ---
C3_BYTE_OFFSET = 13
C4_BYTE_OFFSET = 16

# --- Feature extraction bands (Hz) ---
ALPHA_BAND_HZ = (8, 13)
BETA_BAND_HZ = (13, 30)

# --- Trial timing (seconds), from the experiment protocol ---
ACTIVE_TRIAL_DURATION_S = 4.1
REST_TRIAL_DURATION_S = 4.2

# --- Class labels used throughout the app ---
REST_CLASS = 0
LEFT_HAND_CLASS = 1
RIGHT_HAND_CLASS = 2

# --- Live EEG signal panel ---
SIGNAL_DISPLAY_WINDOW_S = 8
SIGNAL_PLOT_YLIM_UV = 500
