"""Tkinter application: trains the LDA classifier and runs the live motor-imagery test."""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from bci.config import (
    ACTIVE_TRIAL_DURATION_S,
    LEFT_ACTIVE_IMAGE,
    LEFT_HAND_CLASS,
    LEFT_NEUTRAL_IMAGE,
    REST_TRIAL_DURATION_S,
    RIGHT_ACTIVE_IMAGE,
    RIGHT_HAND_CLASS,
    RIGHT_NEUTRAL_IMAGE,
    SAMPLING_RATE_HZ,
    SIGNAL_DISPLAY_WINDOW_S,
    SIGNAL_PLOT_YLIM_UV,
    TEST_INSTRUCTIONS_PATH,
    TRAINING_DATA_PATH,
)
from bci.signal_processing import alpha_beta_band_power, segment_trials_and_extract_features
from bci.smarting_protocol import SmartingClient

# --- Color palette ---
BG_COLOR = "#f5f7fa"
PANEL_BG = "#ffffff"
ACCENT_COLOR = "#2563eb"
ACCENT_DARK = "#1d4ed8"
TEXT_COLOR = "#1e293b"
MUTED_TEXT_COLOR = "#64748b"
BORDER_COLOR = "#e2e8f0"
C3_LINE_COLOR = "#2563eb"
C4_LINE_COLOR = "#f97316"

# --- Fonts ---
LABEL_FONT = ("Segoe UI", 11)
MONO_FONT = ("Consolas", 10)
HEADING_FONT = ("Segoe UI", 16, "bold")
SECTION_FONT = ("Segoe UI", 10, "bold")


class BCIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Motor Imagery BCI")
        self.geometry("1920x1080")
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))
        self.configure(bg=BG_COLOR)

        self._build_style()

        left_panel = ttk.Frame(self, padding=24, style="Card.TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 10), pady=20)

        ttk.Label(left_panel, text="Motor Imagery BCI", style="Heading.Card.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(left_panel, text="Real-time EEG classification", style="Muted.Card.TLabel").pack(anchor="w", pady=(0, 20))

        button_row = ttk.Frame(left_panel, style="Card.TFrame")
        button_row.pack(pady=(0, 20), anchor="w")

        ttk.Button(button_row, text="Train", command=self.train_model, width=12, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Start Test", command=self.start_test, width=12, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Stop Test", command=self.stop_test, width=12, style="Accent.TButton").pack(side=tk.LEFT)

        ttk.Separator(left_panel).pack(fill=tk.X, pady=(0, 18))

        self.ground_truth_var = tk.StringVar(value="Ground Truth: -")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.accuracy_var = tk.StringVar(value="Accuracy: -")

        ttk.Label(left_panel, text="RESULTS", style="Section.Card.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(left_panel, textvariable=self.ground_truth_var, style="Mono.Card.TLabel", wraplength=280, justify="left").pack(anchor="w", pady=3)
        ttk.Label(left_panel, textvariable=self.prediction_var, style="Mono.Card.TLabel", wraplength=280, justify="left").pack(anchor="w", pady=3)
        ttk.Label(left_panel, textvariable=self.accuracy_var, style="Accuracy.Card.TLabel").pack(anchor="w", pady=(12, 0))

        right_panel = ttk.Frame(self, padding=(10, 20, 20, 20))
        right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.figure = Figure(figsize=(9, 7), dpi=100, facecolor=PANEL_BG)
        grid = self.figure.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.15)
        self.signal_axis = self.figure.add_subplot(grid[0, :])
        self.left_axis = self.figure.add_subplot(grid[1, 0])
        self.right_axis = self.figure.add_subplot(grid[1, 1])

        self._style_axis(self.signal_axis)
        self.signal_axis.set_title("Live EEG Signal (C3 / C4)", fontsize=11, color=TEXT_COLOR, fontweight="bold")
        self.signal_axis.set_xlabel("Time [s]", fontsize=9, color=MUTED_TEXT_COLOR)
        self.signal_axis.set_ylabel("Amplitude [µV]", fontsize=9, color=MUTED_TEXT_COLOR)
        self.signal_axis.set_ylim(-SIGNAL_PLOT_YLIM_UV, SIGNAL_PLOT_YLIM_UV)

        self.signal_length = int(SIGNAL_DISPLAY_WINDOW_S * SAMPLING_RATE_HZ)
        self.time_axis = np.linspace(-SIGNAL_DISPLAY_WINDOW_S, 0, self.signal_length)
        self.c3_display = np.zeros(self.signal_length)
        self.c4_display = np.zeros(self.signal_length)

        (self.c3_line,) = self.signal_axis.plot(self.time_axis, self.c3_display, color=C3_LINE_COLOR, linewidth=1.2, label="C3")
        (self.c4_line,) = self.signal_axis.plot(self.time_axis, self.c4_display, color=C4_LINE_COLOR, linewidth=1.2, label="C4")
        self.signal_axis.set_xlim(self.time_axis[0], self.time_axis[-1])
        self.signal_axis.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT_COLOR)

        self.left_neutral_image = mpimg.imread(LEFT_NEUTRAL_IMAGE)
        self.right_neutral_image = mpimg.imread(RIGHT_NEUTRAL_IMAGE)
        self.left_active_image = mpimg.imread(LEFT_ACTIVE_IMAGE)
        self.right_active_image = mpimg.imread(RIGHT_ACTIVE_IMAGE)

        self._style_axis(self.left_axis)
        self.left_axis.imshow(self.left_neutral_image)
        self.left_axis.axis("off")
        self.left_axis.set_title("Left", fontsize=11, color=TEXT_COLOR, fontweight="bold")

        self._style_axis(self.right_axis)
        self.right_axis.imshow(self.right_neutral_image)
        self.right_axis.axis("off")
        self.right_axis.set_title("Right", fontsize=11, color=TEXT_COLOR, fontweight="bold")

        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        self.lda_model = None
        self.ground_truth = []
        self.predictions = []
        self.trial_count = 0
        self.test_instructions = pd.read_csv(TEST_INSTRUCTIONS_PATH, header=0)
        self.current_instruction = int(self.test_instructions.iloc[self.trial_count, 0])
        self.channel_c3_buffer = []
        self.channel_c4_buffer = []

        try:
            self.smarting_client = SmartingClient()
        except serial.SerialException as exc:
            self.smarting_client = None
            messagebox.showerror(
                "Connection Error",
                f"Could not open the SMARTING serial connection:\n{exc}\n\n"
                "You can still train the model. Connect the device and restart "
                "the app to run a live test.",
            )

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=BG_COLOR)
        style.configure("Card.TFrame", background=PANEL_BG, relief="flat")

        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=LABEL_FONT)
        style.configure("Card.TLabel", background=PANEL_BG, foreground=TEXT_COLOR, font=LABEL_FONT)
        style.configure("Heading.Card.TLabel", background=PANEL_BG, foreground=TEXT_COLOR, font=HEADING_FONT)
        style.configure("Muted.Card.TLabel", background=PANEL_BG, foreground=MUTED_TEXT_COLOR, font=LABEL_FONT)
        style.configure("Section.Card.TLabel", background=PANEL_BG, foreground=MUTED_TEXT_COLOR, font=SECTION_FONT)
        style.configure("Mono.Card.TLabel", background=PANEL_BG, foreground=TEXT_COLOR, font=MONO_FONT)
        style.configure("Accuracy.Card.TLabel", background=PANEL_BG, foreground=ACCENT_COLOR, font=("Segoe UI", 13, "bold"))

        style.configure(
            "Accent.TButton",
            font=LABEL_FONT,
            padding=8,
            background=ACCENT_COLOR,
            foreground="white",
            borderwidth=0,
            focuscolor=ACCENT_COLOR,
        )
        style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK)])

    @staticmethod
    def _style_axis(axis):
        axis.set_facecolor(PANEL_BG)
        for spine in axis.spines.values():
            spine.set_color(BORDER_COLOR)
        axis.tick_params(colors=MUTED_TEXT_COLOR, labelsize=8)

    def start_test(self):
        if self.smarting_client is None:
            messagebox.showerror(
                "Connection Error",
                "No SMARTING connection is available. Connect the device and restart the app.",
            )
            return
        self.smarting_client.start_acquisition()
        self.update_plot()

    def update_plot(self):
        for c3_sample, c4_sample in self.smarting_client.read_c3_c4_samples():
            self.channel_c3_buffer.append(c3_sample)
            self.channel_c4_buffer.append(c4_sample)

            self.c3_display = np.roll(self.c3_display, -1)
            self.c3_display[-1] = c3_sample
            self.c4_display = np.roll(self.c4_display, -1)
            self.c4_display[-1] = c4_sample

            if len(self.channel_c3_buffer) >= ACTIVE_TRIAL_DURATION_S * SAMPLING_RATE_HZ and self.trial_count % 2 != 0:
                c3_alpha, c3_beta = alpha_beta_band_power(self.channel_c3_buffer)
                c4_alpha, c4_beta = alpha_beta_band_power(self.channel_c4_buffer)
                feature_vector = [[c3_alpha, c3_beta, c4_alpha, c4_beta]]

                prediction = self.lda_model.predict(feature_vector)[0]
                self.ground_truth.append(self.current_instruction)
                self.predictions.append(prediction.item())
                self.ground_truth_var.set("Ground Truth: " + str(self.ground_truth))
                self.prediction_var.set("Prediction: " + str(self.predictions))

                running_accuracy = accuracy_score(self.ground_truth, self.predictions)
                self.accuracy_var.set(f"Accuracy: {running_accuracy * 100:.2f}%")

                self.channel_c3_buffer, self.channel_c4_buffer = [], []
                self.trial_count += 1

                self.left_axis.imshow(self.left_neutral_image)
                self.right_axis.imshow(self.right_neutral_image)

                if self.trial_count <= 35:
                    self.current_instruction = int(self.test_instructions.iloc[self.trial_count, 0])

            elif len(self.channel_c3_buffer) >= REST_TRIAL_DURATION_S * SAMPLING_RATE_HZ and self.trial_count % 2 == 0:
                self.channel_c3_buffer, self.channel_c4_buffer = [], []
                self.trial_count += 1

                if self.trial_count <= 35:
                    self.current_instruction = int(self.test_instructions.iloc[self.trial_count, 0])

                if self.current_instruction == LEFT_HAND_CLASS:
                    self.left_axis.imshow(self.left_active_image)
                    self.right_axis.imshow(self.right_neutral_image)
                elif self.current_instruction == RIGHT_HAND_CLASS:
                    self.left_axis.imshow(self.left_neutral_image)
                    self.right_axis.imshow(self.right_active_image)

        self.c3_line.set_ydata(self.c3_display)
        self.c4_line.set_ydata(self.c4_display)
        self.canvas.draw_idle()

        self.after(100, self.update_plot)

    def stop_test(self):
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        self.accuracy_var.set(f"Accuracy: {accuracy * 100:.2f}%")
        if self.smarting_client is not None:
            self.smarting_client.stop_acquisition()
            self.smarting_client.close()
        self._show_confusion_matrix()

    def _show_confusion_matrix(self):
        if not self.ground_truth:
            return

        matrix = confusion_matrix(self.ground_truth, self.predictions, labels=[LEFT_HAND_CLASS, RIGHT_HAND_CLASS])
        display = ConfusionMatrixDisplay(matrix, display_labels=["Left", "Right"])

        window = tk.Toplevel(self)
        window.title("Test Results — Confusion Matrix")
        window.configure(bg=PANEL_BG)

        figure = Figure(figsize=(5, 5), dpi=100, facecolor=PANEL_BG)
        axis = figure.add_subplot(1, 1, 1)
        axis.set_facecolor(PANEL_BG)

        display.plot(ax=axis, cmap="Blues", colorbar=False)

        axis.set_title("Confusion Matrix", fontsize=12, color=TEXT_COLOR, fontweight="bold")
        axis.xaxis.label.set_color(TEXT_COLOR)
        axis.yaxis.label.set_color(TEXT_COLOR)
        axis.tick_params(colors=TEXT_COLOR)
        for spine in axis.spines.values():
            spine.set_color(BORDER_COLOR)

        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=20, pady=20)

    def train_model(self):
        data = pd.read_csv(TRAINING_DATA_PATH, header=0)
        channel_c3 = data.iloc[:, 4]
        channel_c4 = data.iloc[:, 5]
        labels = data.iloc[:, -1]
        features, trial_labels = segment_trials_and_extract_features(labels, channel_c3, channel_c4)
        print(len(features))
        print(len(trial_labels))
        lda = LinearDiscriminantAnalysis()
        self.lda_model = lda.fit(features, trial_labels)
