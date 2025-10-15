import mss
from PIL import Image
import pytesseract
from pynput import keyboard
import random
import time
from datetime import datetime
import threading
import sys
import os
from twocaptcha import TwoCaptcha
from twocaptcha import ApiException
from src.CooldownGenerator.cooldown_humanizer import HumanCooldown
import tkinter as tk
from tkinter import ttk

# =====================================================================
# --- 1. CONFIGURATION ---
# =====================================================================

# BOT SETTINGS
BASE_SLEEP = 3.0
humanizedCooldown = HumanCooldown(base=BASE_SLEEP)
LOG_FILE = None  # Defined in __main__

# API CONFIG
# **IMPORTANT**: Replace with your 2Captcha API key. Uses environment variable as fallback.
API_KEY = os.getenv('APIKEY_2CAPTCHA', '6840474409ea1ab7504ed66ea906ba7d')

# SCREEN CAPTURE AREA (Must be defined to capture the chat window containing the CAPTCHA)
CAPTURE_AREA = {
    'left': 650,  # X-coordinate of the left edge
    'top': 150,  # Y-coordinate of the top edge
    'width': 600,  # Width of the area
    'height': 750  # Height of the area
}

# TESSERACT CONFIG
# **IMPORTANT**: Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# CAPTCHA HANDLING CONSTANTS
ANTI_BOT_DISABLE_DURATION = 10 * 60  # 10 minutes
POST_CAPTCHA_DELAY_RANGE = (4, 10)  # Seconds

# =====================================================================
# --- 2. GLOBAL STATE AND THREAD CONTROLS ---
# =====================================================================

running = False
fish_thread = None
check_thread = None
solver = None
captcha_found = threading.Event()
paused = threading.Event()
TERMINATION_REASON = "Script manually started/stopped."
LAST_SOLVED_CODE = None
ANTI_BOT_DISABLED_UNTIL = 0.0
LAST_SUPPRESSION_LOG = 0.0

# =====================================================================
# --- 3. LOGGING AND CONTROL FUNCTIONS ---
# =====================================================================

def log_event(level, function_name, message, ocr_text=None):
    """Logs a runtime event to the console (concise) and file (detailed)."""
    global LOG_FILE

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # --- 1. Console Message (No OCR Text) ---
    console_message = f"[{timestamp}] [{level:<7}] [{function_name:<20}] {message}"
    print(console_message)

    # --- 2. File Message (Includes OCR Text) ---
    file_message = console_message + "\n"

    if ocr_text:
        # Indent the OCR text block for readability in the log file
        ocr_lines = "\n".join([f"  [OCR_TEXT] {line}" for line in ocr_text.splitlines() if line.strip()])
        file_message += f"\n{ocr_lines}"

    file_message += "\n"

    try:
        with open(LOG_FILE, 'a') as f:
            f.write(file_message)
    except Exception:
        # Avoid recursive logging errors if file write fails
        pass


def log_termination(reason, details=""):
    """Logs the script stop event and final state."""

    log_event("TERMINATE", "stop_script", f"TERMINATED: {reason}")

    # Add final state summary to the log file for diagnostics
    summary = (
        f"  Reason: {reason}\n"
        f"  Details: {details}\n"
        f"  Active CAPTCHA State: {captcha_found.is_set()}\n"
        f"  Last Captcha Solution: {LAST_SOLVED_CODE}\n"
        f"----------------------------------------\n"
    )

    try:
        with open(LOG_FILE, 'a') as f:
            f.write(summary)
    except Exception:
        pass

    print(f"\n[LOGGER] Termination reason saved to {LOG_FILE}.")


def stop_script(reason="Unknown Error", details="", exit_program=False):
    """Gracefully stops all loops, logs the reason, and optionally exits the program."""
    global running
    global TERMINATION_REASON

    if not running:
        print("Script is already stopped.")
        return

    TERMINATION_REASON = reason
    print(f"\n--- Stopping Bot ({reason}) ---\n")
    running = False
    paused.clear()
    captcha_found.set()  # Release any waiting threads

    time.sleep(1)
    log_termination(TERMINATION_REASON, details)

    if exit_program:
        sys.exit(0)


# =====================================================================
# --- 4. CORE UTILITY FUNCTIONS ---
# =====================================================================


def pause_script():
    """Pauses the fish and check loops without terminating them."""
    if not running:
        print("Cannot pause: script is not running.")
        return

    if paused.is_set():
        print("Script is already paused.")
        return

    paused.set()
    log_event("PAUSE", "pause_script", "Bot execution paused by user.")


def resume_script():
    """Resumes the fish and check loops after a pause."""
    if not running:
        print("Cannot resume: script is not running.")
        return

    if not paused.is_set():
        print("Script is not paused.")
        return

    paused.clear()
    log_event("RESUME", "resume_script", "Bot execution resumed by user.")


def update_base_cooldown(new_base):
    """Updates the base cooldown at runtime and refreshes the cooldown generator."""
    global BASE_SLEEP
    global humanizedCooldown

    try:
        new_value = float(new_base)
    except (TypeError, ValueError):
        print("Invalid cooldown value. Please provide a numeric input.")
        return False

    if new_value <= 0:
        print("Cooldown value must be greater than zero.")
        return False

    BASE_SLEEP = new_value

    try:
        humanizedCooldown.retune_base(new_value)
    except AttributeError:
        humanizedCooldown.base = new_value
    except ValueError as exc:
        print(f"Cooldown update error: {exc}")
        return False

    log_event("CONFIG", "update_base_cooldown", f"Base cooldown adjusted to {new_value:.2f}s.")
    return True

def save_screenshot(filename):
    """Captures and saves a screenshot of the CAPTURE_AREA to the specified filename."""
    try:
        with mss.mss() as sct:
            sct_img = sct.grab(CAPTURE_AREA)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(filename)
            return img
    except Exception as e:
        log_event("ERROR", "save_screenshot", f"Could not capture screen: {e}")
        return None


def save_last_captcha_image(filename="api_captcha_current.png"):
    """
    Saves the latest screenshot for the 2Captcha API.
    Uses a static name to overwrite previous images.
    Returns the file path.
    """
    img = save_screenshot(filename)
    if img is not None:
        log_event("INFO", "save_last_captcha_image", f"Saved latest API image to: {filename}")
        return filename
    return None


def ocr_screenshot(img):
    """Performs Tesseract OCR on the given PIL image."""
    try:
        raw_text = pytesseract.image_to_string(img, config='--psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-').strip()
        return raw_text
    except Exception as e:
        log_event("ERROR", "ocr_screenshot", f"Tesseract OCR failed: {e}")
        return ""


def keyboard_input(text, delay=0.01):
    controller = keyboard.Controller()
    for char in text:
        controller.press(char)
        controller.release(char)
        time.sleep(delay)


def keyboard_press(key):
    controller = keyboard.Controller()
    controller.press(key)
    controller.release(key)


def clear_input_line():
    controller = keyboard.Controller()
    controller.press(keyboard.Key.ctrl)
    controller.press('a')
    controller.release('a')
    controller.release(keyboard.Key.ctrl)
    time.sleep(0.05)
    keyboard_press(keyboard.Key.backspace)
    time.sleep(0.1)


# =====================================================================
# --- 5. CAPTCHA VERIFICATION HELPERS ---
# =====================================================================

def api_solve_captcha(img_path):
    """Sends the CAPTCHA image path to the 2Captcha service for solving."""
    if solver is None:
        log_event("ERROR", "api_solve_captcha", "Solver not initialized. Cannot use API.")
        return None

    try:
        log_event("API_CALL", "api_solve_captcha", f"Sending image path {img_path} to 2Captcha service...")

        result = solver.normal(
            file=img_path,
            caseSensitive=1,
            minLen=4,
            maxLen=8
        )

        captcha_text = result['code']
        log_event("API_CALL", "api_solve_captcha", f"Received solution: **{captcha_text}**")
        return captcha_text

    except ApiException as e:
        log_event("API_FAIL", "api_solve_captcha", f"2Captcha API failed: {e}")
        return None
    except Exception as e:
        log_event("ERROR", "api_solve_captcha", f"An unexpected API error occurred: {e}")
        return None


def verify_api_response_timing(code):
    """
    Sends /verify code, then uses the complex 35s/10s timing (API Path).
    Returns True (success and resume) or False (stop script).
    """

    log_event("COMMAND", "verify_api_response_timing", f"Inputting API result: /verify {code}")
    clear_input_line()
    time.sleep(0.5)
    keyboard_input("/verify")
    time.sleep(0.1)
    keyboard_press(keyboard.Key.enter)
    keyboard_input(code)
    time.sleep(0.1)
    keyboard_press(keyboard.Key.enter)

    # --- Check 1: Wait 35 seconds ---
    log_event("INFO", "verify_api_response_timing", "Waiting 35s for API verification result.")
    time.sleep(35.0)

    for attempt in range(1, 3):
        img = save_screenshot("verification_check.png")
        raw_text = ocr_screenshot(img)
        raw_text_lower = raw_text.lower()
        log_event("VERIFY", "verify_api_response_timing", f"Screen check {attempt} after wait.", raw_text)

        if "may now continue" in raw_text_lower:
            log_event("SUCCESS", "verify_api_response_timing", "API code accepted. Resuming loop.")
            return True

        elif "incorrect" in raw_text_lower:
            log_event("FAILURE", "verify_api_response_timing", "API code was incorrect.")
            return False

        if attempt == 1:
            log_event("INFO", "verify_api_response_timing", "Result unclear. Waiting 10s more.")
            time.sleep(10.0)
        else:
            log_event("CRITICAL", "verify_api_response_timing",
                      "Verification failed to get clear response after 45 seconds.")
            return False

    return False
# =====================================================================
# --- 6. MAIN THREAD LOOPS ---
# =====================================================================

def fish_loop():
    """Main loop: continuously inputs /fish command."""
    global running
    global humanizedCooldown

    while running:
        if paused.is_set():
            log_event("INFO", "fish_loop", "Paused. Waiting to resume.")
            while paused.is_set() and running:
                time.sleep(0.2)
            if not running:
                break

        if not captcha_found.is_set():
            log_event("COMMAND", "fish_loop", "Executing /fish command.")

            keyboard_input("/fish")
            keyboard_press(keyboard.Key.enter)
            keyboard_press(keyboard.Key.enter)

            lag = humanizedCooldown.next()
            log_event("INFO", "fish_loop", f"Sleeping for {lag:.2f} seconds (Cooldown).")

            captcha_found.wait(timeout=lag)
        else:
            captcha_found.wait(timeout=1)

    log_event("INFO", "fish_loop", "Thread stopped.")


def check_loop():
    """Second loop: continuously checks the screen for captcha or stop warnings."""
    global running
    global LAST_SOLVED_CODE
    global ANTI_BOT_DISABLED_UNTIL
    global LAST_SUPPRESSION_LOG

    while running:
        if paused.is_set():
            time.sleep(0.5)
            continue

        time.sleep(1)

        img = save_screenshot("current_screen_check.png")  # Capture screen for OCR
        if img is None:
            continue

        raw_text = ocr_screenshot(img)
        raw_text_lower = raw_text.lower()

        # Log periodic status for diagnostic purposes
        if time.time() % 30 < 1:
            log_event("INFO", "check_loop", "Screen scan (periodic check).", raw_text)

        # --- CAPTCHA DETECTION FLOW ---

        if "anti-bot" in raw_text_lower:
            current_time = time.time()

            if current_time < ANTI_BOT_DISABLED_UNTIL:
                remaining = ANTI_BOT_DISABLED_UNTIL - current_time
                if captcha_found.is_set():
                    captcha_found.clear()
                if current_time - LAST_SUPPRESSION_LOG >= 5:
                    log_event(
                        "INFO",
                        "check_loop",
                        f"Anti-bot detection suppressed for another {remaining:.0f}s after recent success.",
                        raw_text,
                    )
                    LAST_SUPPRESSION_LOG = current_time
                continue

            if not captcha_found.is_set():
                captcha_found.set()
                log_event("CAPTCHA", "check_loop", "CAPTCHA detected. Pausing fish loop.", raw_text)

            last_img_path = save_last_captcha_image()
            if last_img_path is None:
                stop_script("Failed to capture captcha image.", details="Screenshot capture failed.", exit_program=True)
                return

            solved_code = api_solve_captcha(last_img_path)

            if not solved_code:
                stop_script("2Captcha API failed.", details="API call failed or returned no code.", exit_program=True)
                return

            if not verify_api_response_timing(solved_code):
                stop_script("API result verification failed.", details=f"API Code: {solved_code}", exit_program=True)
                return

            LAST_SOLVED_CODE = solved_code
            ANTI_BOT_DISABLED_UNTIL = time.time() + ANTI_BOT_DISABLE_DURATION
            LAST_SUPPRESSION_LOG = 0.0

            delay = random.uniform(*POST_CAPTCHA_DELAY_RANGE)
            disable_minutes = ANTI_BOT_DISABLE_DURATION / 60
            log_event(
                "SUCCESS",
                "check_loop",
                (
                    f"Captcha solved via 2Captcha ({solved_code}). Waiting {delay:.2f}s before resuming. "
                    f"Anti-bot detection disabled for {disable_minutes:.1f} minutes."
                ),
            )

            time.sleep(delay)
            captcha_found.clear()

        else:
            if captcha_found.is_set():
                captcha_found.clear()

    log_event("INFO", "check_loop", "Thread stopped.")

# =====================================================================
# --- 7. DEBUGGING AND LISTENER SETUP ---
# =====================================================================

def test_scan_for_captcha():
    """F10: Captures screen, performs OCR, and tests API functionality."""
    global solver
    print("\n*** F10 Pressed! Starting CAPTCHA Test Scan... ***")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"test_screenshot_{timestamp}.png"

    img = save_screenshot(screenshot_filename)
    if img is None:
        return

    raw_text = ocr_screenshot(img)
    log_event("TEST", "test_scan_for_captcha", "F10 Test Scan performed.", raw_text)

    print("\n--- OCR Raw Result ---")
    print(raw_text)
    print("----------------------")

    if "anti-bot" in raw_text.lower():
        print("Detected 'Anti-bot' string during test scan. Script will use 2Captcha for resolution.")

    if solver is None:
        print("ERROR: 2Captcha solver is not initialized. Cannot test API.")
        return

    print("Solving captcha with 2Captcha API...")
    try:
        result = solver.normal(file=screenshot_filename, caseSensitive=1)
        print("\n--- 2CAPTCHA API Result ---")
        print(f"Solved Code: **{result['code']}**")
        print("---------------------------")
    except Exception as e:
        print(f"ERROR: Could not solve captcha: {e}")


def start_script():
    """Initializes and starts both fish and check threads on F8."""
    global running
    global fish_thread
    global check_thread
    global LAST_SOLVED_CODE
    global ANTI_BOT_DISABLED_UNTIL

    if running:
        print("Script is already running.")
        return

    running = True
    paused.clear()
    captcha_found.clear()
    LAST_SOLVED_CODE = None
    ANTI_BOT_DISABLED_UNTIL = 0.0
    LAST_SUPPRESSION_LOG = 0.0
    log_event("START", "start_script", "Bot threads initialized and started.")

    print("\n--- Starting Bot (Press F9 to stop) ---")

    check_thread = threading.Thread(target=check_loop, daemon=True)
    check_thread.start()

    time.sleep(0.5)

    fish_thread = threading.Thread(target=fish_loop, daemon=True)
    fish_thread.start()

    print("Bot is running. Ensure Discord is the active window.")


def on_press(key):
    """Handles key presses. F8 to start, F9 to stop, F10 to test scan."""
    try:
        if key == keyboard.Key.f8:
            start_script()
        elif key == keyboard.Key.f9:
            stop_script("Script stopped manually by F9 key.")
        elif key == keyboard.Key.f10:
            test_scan_for_captcha()
    except AttributeError:
        pass


def listen_for_f_keys():
    """Sets up the keyboard listener for F8, F9, and F10."""
    print("---------------------------------------")
    print("Discord Fishing Bot v3.6 (No Suppression Logic)")
    print("F8: START | F9: STOP | F10: TEST OCR/API")
    print(f"Log File: {LOG_FILE}")
    print("---------------------------------------")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


class BotControlGUI(tk.Tk):
    """Graphical control panel for managing the fishing bot."""

    def __init__(self):
        super().__init__()

        # --- Window chrome / overlay behavior ---
        self.overrideredirect(True)
        self.wm_attributes("-topmost", True)
        self.wm_attributes("-alpha", 0.92)
        self.configure(bg="#111827")
        self.geometry("420x260")
        self.resizable(False, False)

        self.after(50, self.release_focus)

        self._drag_offset = (0, 0)
        self._scale_commit_job = None

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#111827")
        style.configure("Overlay.TFrame", background="#1f2933")
        style.configure("OverlayAccent.TFrame", background="#111827")
        style.configure("Title.TLabel", background="#111827", foreground="#f5f7fa", font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", background="#1f2933", foreground="#e4e8f0", font=("Segoe UI", 12, "bold"))
        style.configure("Detail.TLabel", background="#1f2933", foreground="#cbd2d9", font=("Segoe UI", 10))
        style.configure(
            "Control.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            background="#334155",
            foreground="#f5f7fa",
        )
        style.map(
            "Control.TButton",
            background=[("active", "#3e4c59"), ("!active", "#334155")],
            foreground=[("disabled", "#7b8794"), ("!disabled", "#f5f7fa")],
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            background="#2563eb",
            foreground="#f5f7fa",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("!active", "#2563eb")],
            foreground=[("disabled", "#9aa5b1"), ("!disabled", "#f5f7fa")],
        )

        chrome = ttk.Frame(self, style="OverlayAccent.TFrame", padding=(12, 8))
        chrome.pack(fill=tk.X)

        title = ttk.Label(chrome, text="Fishing Bot Overlay", style="Title.TLabel")
        title.pack(side=tk.LEFT)

        close_btn = ttk.Button(
            chrome,
            text="✕",
            width=3,
            style="Control.TButton",
            command=self.close_overlay,
        )
        close_btn.pack(side=tk.RIGHT)

        chrome.bind("<ButtonPress-1>", self.start_move)
        chrome.bind("<B1-Motion>", self.do_move)
        chrome.bind("<ButtonRelease-1>", self.stop_move)
        title.bind("<ButtonPress-1>", self.start_move)
        title.bind("<B1-Motion>", self.do_move)
        title.bind("<ButtonRelease-1>", self.stop_move)

        body = ttk.Frame(self, style="Overlay.TFrame", padding=15)
        body.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(body, style="Overlay.TFrame", padding=10)
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel").pack(anchor=tk.W)

        self.detail_var = tk.StringVar(value=f"Base cooldown: {BASE_SLEEP:.2f}s")
        ttk.Label(status_frame, textvariable=self.detail_var, style="Detail.TLabel").pack(anchor=tk.W, pady=(4, 0))

        self.feedback_var = tk.StringVar(value="")
        self.feedback_label = ttk.Label(status_frame, textvariable=self.feedback_var, style="Detail.TLabel")
        self.feedback_label.pack(anchor=tk.W, pady=(4, 0))

        button_frame = ttk.Frame(body, style="Overlay.TFrame")
        button_frame.pack(fill=tk.X, pady=(12, 4))

        self.start_button = ttk.Button(
            button_frame,
            text="Start",
            style="Accent.TButton",
            command=self.handle_start,
        )
        self.start_button.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.pause_button = ttk.Button(
            button_frame,
            text="Pause",
            style="Control.TButton",
            command=self.toggle_pause,
        )
        self.pause_button.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            style="Control.TButton",
            command=self.handle_stop,
        )
        self.stop_button.grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        button_frame.columnconfigure((0, 1, 2), weight=1)

        cooldown_frame = ttk.Frame(body, style="Overlay.TFrame", padding=(0, 8, 0, 0))
        cooldown_frame.pack(fill=tk.X)

        ttk.Label(cooldown_frame, text="Cooldown (seconds)", style="Detail.TLabel").grid(row=0, column=0, sticky="w")

        self.cooldown_value = tk.DoubleVar(value=BASE_SLEEP)
        self.cooldown_entry_var = tk.StringVar(value=f"{BASE_SLEEP:.2f}")

        self.cooldown_entry = ttk.Entry(cooldown_frame, textvariable=self.cooldown_entry_var, font=("Segoe UI", 10))
        self.cooldown_entry.grid(row=0, column=1, padx=(8, 4), sticky="ew")
        self.cooldown_entry.bind("<Return>", self.apply_cooldown)
        self.cooldown_entry.bind("<FocusOut>", self.apply_cooldown)

        self.cooldown_apply = ttk.Button(
            cooldown_frame,
            text="Set",
            style="Control.TButton",
            command=self.apply_cooldown,
        )
        self.cooldown_apply.grid(row=0, column=2, padx=(4, 0))

        self.cooldown_scale = ttk.Scale(
            cooldown_frame,
            variable=self.cooldown_value,
            from_=0.2,
            to=30.0,
            orient=tk.HORIZONTAL,
            command=self.on_cooldown_scale,
        )
        self.cooldown_scale.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        cooldown_frame.columnconfigure(1, weight=1)

        self.start_button.state(["!disabled"])
        self.pause_button.state(["disabled"])
        self.stop_button.state(["disabled"])

        self.after(250, self.poll_state)

    def handle_start(self):
        if solver is None:
            self.show_feedback("2Captcha solver not initialized.")
            return

        start_script()
        self.show_feedback("Bot started.")
        self.release_focus()
        self.poll_state(force=True)

    def handle_stop(self):
        if running:
            stop_script("Script stopped via overlay.")
            self.show_feedback("Bot stopped.")
        else:
            self.show_feedback("Bot already stopped.")
        self.release_focus()
        self.poll_state(force=True)

    def toggle_pause(self):
        if not running:
            self.show_feedback("Start the bot before pausing.")
            return

        if paused.is_set():
            resume_script()
            self.show_feedback("Resumed.")
        else:
            pause_script()
            self.show_feedback("Paused.")
        self.release_focus()
        self.poll_state(force=True)

    def apply_cooldown(self, event=None):
        value = self.cooldown_entry_var.get()
        if update_base_cooldown(value):
            self.cooldown_entry_var.set(f"{BASE_SLEEP:.2f}")
            self.cooldown_value.set(BASE_SLEEP)
            self.detail_var.set(f"Base cooldown: {BASE_SLEEP:.2f}s")
            self.show_feedback(f"Cooldown set to {BASE_SLEEP:.2f}s")
        else:
            self.cooldown_entry_var.set(f"{BASE_SLEEP:.2f}")
            self.show_feedback("Enter a positive number.")

        if event is None or getattr(event, "keysym", "") == "Return":
            self.release_focus()

        if event is not None and getattr(event, "keysym", "") == "Return":
            return "break"

    def on_cooldown_scale(self, value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return

        self.cooldown_entry_var.set(f"{numeric_value:.2f}")

        if self._scale_commit_job:
            self.after_cancel(self._scale_commit_job)
        self._scale_commit_job = self.after(150, self.commit_scale_value)

    def commit_scale_value(self):
        self._scale_commit_job = None
        value = self.cooldown_value.get()
        if update_base_cooldown(value):
            self.cooldown_entry_var.set(f"{BASE_SLEEP:.2f}")
            self.detail_var.set(f"Base cooldown: {BASE_SLEEP:.2f}s")
            self.show_feedback(f"Cooldown set to {BASE_SLEEP:.2f}s")
            self.release_focus()

    def show_feedback(self, message, duration=2500):
        self.feedback_var.set(message)
        if hasattr(self, "_feedback_job") and self._feedback_job is not None:
            self.after_cancel(self._feedback_job)
        self._feedback_job = self.after(duration, self.clear_feedback)

    def clear_feedback(self):
        self.feedback_var.set("")
        self._feedback_job = None

    def start_move(self, event):
        self._drag_offset = (event.x_root - self.winfo_rootx(), event.y_root - self.winfo_rooty())

    def stop_move(self, _event):
        self._drag_offset = (0, 0)

    def do_move(self, event):
        if self._drag_offset == (0, 0):
            return
        x = event.x_root - self._drag_offset[0]
        y = event.y_root - self._drag_offset[1]
        self.geometry(f"+{x}+{y}")

    def close_overlay(self):
        if running:
            stop_script("Overlay closed by user.")
        self.destroy()

    def poll_state(self, force=False):
        state_text = "Idle"
        if running:
            state_text = "Paused" if paused.is_set() else "Running"
        self.status_var.set(f"Status: {state_text}")

        if self.cooldown_entry.focus_get() is not self.cooldown_entry or force:
            self.cooldown_entry_var.set(f"{BASE_SLEEP:.2f}")
            self.cooldown_value.set(BASE_SLEEP)

        self.detail_var.set(f"Base cooldown: {BASE_SLEEP:.2f}s")

        if running:
            self.start_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            self.pause_button.state(["!disabled"])
            self.pause_button.configure(text="Resume" if paused.is_set() else "Pause")
        else:
            self.start_button.state(["!disabled"])
            self.stop_button.state(["disabled"])
            self.pause_button.state(["disabled"])
            self.pause_button.configure(text="Pause")

        self.after(250, self.poll_state)

    def release_focus(self):
        if not self.winfo_exists():
            return

        def _drop_focus():
            if not self.winfo_exists():
                return
            try:
                self.tk.call("focus", "none")
            except tk.TclError:
                pass

        self.after(60, _drop_focus)

if __name__ == "__main__":
    # Initialize the unique log file name on script start
    current_date = datetime.now().strftime('%Y%m%d')
    i = 0
    # Ensure the log file is unique for today's run
    while os.path.exists(f"{current_date}_bot_log_{i}.txt"):
        i += 1
    LOG_FILE = f"{current_date}_bot_log_{i}.txt"

    # Initialize the 2Captcha solver once on startup
    try:
        solver = TwoCaptcha(API_KEY)
        print("2Captcha Solver initialized.")
    except Exception as e:
        print(f"!!! CRITICAL: Failed to initialize 2Captcha solver: {e} !!!")
        solver = None

    gui = BotControlGUI()

    listener_thread = threading.Thread(target=listen_for_f_keys, daemon=True)
    listener_thread.start()

    gui.mainloop()
