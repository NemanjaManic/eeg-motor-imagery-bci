"""Entry point for the EEG Motor Imagery BCI application."""

from bci.app import BCIApp

if __name__ == "__main__":
    app = BCIApp()
    app.mainloop()
