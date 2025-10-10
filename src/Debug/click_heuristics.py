import tkinter as tk
from datetime import datetime
import os

# --- Configuration ---
LOG_FILENAME = "click_timestamps.txt"
SOUND_INTERVAL_MS = 3000  # 3000 milliseconds = 3 seconds
BUTTON_COLOR_NORMAL = "#4CAF50"  # Green
BUTTON_COLOR_ALERT = "#FF5733"  # Red-Orange for the 'sound' alert


class ClickTimerApp:
    """
    A simple Tkinter application to log click timestamps and provide a
    recurring visual 'sound' alert.
    """

    def __init__(self, master):
        self.master = master
        master.title("Click Timing App")
        master.geometry("400x300")
        master.configure(bg="#f0f0f0")

        # Initialize log file
        self._initialize_log_file()

        # Create the main clickable box
        self.click_button = tk.Button(
            master,
            text="CLICK ME",
            command=self.log_click,
            bg=BUTTON_COLOR_NORMAL,
            fg="white",
            font=("Inter", 16, "bold"),
            relief=tk.RAISED,
            bd=5,
            activebackground="#45a049"
        )
        self.click_button.pack(expand=True, padx=50, pady=50, fill="both")

        # Information label
        self.info_label = tk.Label(
            master,
            text=f"Logs saved to: {LOG_FILENAME}\nClick the box to record a timestamp.",
            bg="#f0f0f0",
            font=("Inter", 10)
        )
        self.info_label.pack(pady=(0, 10))

        # Start the recurring "sound" function
        self.alert_id = self.master.after(SOUND_INTERVAL_MS, self._alert_handler)

    def _initialize_log_file(self):
        """Creates the log file and writes a header."""
        mode = 'a' if os.path.exists(LOG_FILENAME) else 'w'
        with open(LOG_FILENAME, mode) as f:
            f.write(f"\n--- SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            print(f"Log file initialized: {LOG_FILENAME}")

    def log_click(self):
        """
        Records the current timestamp to the log file and updates the console.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate microseconds for cleaner output

        log_entry = f"CLICK: {timestamp_str}\n"

        # 1. Log to file
        try:
            with open(LOG_FILENAME, "a") as f:
                f.write(log_entry)
        except IOError as e:
            print(f"Error writing to file: {e}")
            self.info_label.config(text=f"ERROR: Could not write to {LOG_FILENAME}")
            return

        # 2. Log to console
        print(log_entry.strip())

        # 3. Provide visual feedback (brief flash)
        self.click_button.config(bg="#FFA500")  # Brief orange flash
        self.master.after(100, lambda: self.click_button.config(bg=BUTTON_COLOR_NORMAL))

    def _alert_handler(self):
        """
        Handles the recurring 3-second 'sound' alert.
        This uses a visual flash and console print to simulate a non-blocking sound.
        """

        # 1. Visual 'Sound' Cue (Button Flash)
        self.click_button.config(bg=BUTTON_COLOR_ALERT)
        self.master.after(200, lambda: self.click_button.config(bg=BUTTON_COLOR_NORMAL))

        # 2. Console 'Sound' Cue (Print)
        print("--- 3 SECOND ALERT ---")

        # 3. Schedule the next alert
        self.alert_id = self.master.after(SOUND_INTERVAL_MS, self._alert_handler)


if __name__ == "__main__":
    # Initialize the main window
    root = tk.Tk()

    # Create and run the application
    app = ClickTimerApp(root)

    # Start the Tkinter event loop
    root.mainloop()

# Note: The log file is automatically closed when the 'with open()' block finishes.
