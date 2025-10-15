import mss
from PIL import Image
import pytesseract
from pytesseract import Output
from pynput import keyboard
import random
import time
from datetime import datetime
import threading
import sys
import os
from src.CooldownGenerator.cooldown_humanizer import HumanCooldown
import tkinter as tk
from tkinter import ttk, messagebox
from anticaptchaofficial.imagecaptcha import imagecaptcha

# =====================================================================
# --- 1. CONFIGURATION ---
# =====================================================================

# BOT SETTINGS
BASE_SLEEP = 3.0
humanizedCooldown = HumanCooldown(base=BASE_SLEEP)
LOG_FILE = None  # Defined in __main__

# SCREEN CAPTURE AREA (Must be defined to capture the chat window containing the CAPTCHA)
CAPTURE_AREA = {
    'left': 350,  # X-coordinate of the left edge
    'top': 150,  # Y-coordinate of the top edge
    'width': 600,  # Width of the area
    'height': 860  # Height of the area
}

# CAPTCHA ANCHOR OFFSETS (relative to the detected 'anti-bot' text)
CAPTCHA_ANCHOR_VERTICAL_OFFSET = 117   # Pixels below the bottom of the anchor to begin the crop
CAPTCHA_ANCHOR_HORIZONTAL_ADJUST = 42  # Horizontal adjustment from the anchor center
CAPTCHA_ANCHOR_CROP_WIDTH = 150       # Width of the captcha crop
CAPTCHA_ANCHOR_CROP_HEIGHT = 32      # Height of the captcha crop

# CAPTCHA IMAGE STORAGE (for manual inspection and API submission)
CAPTCHA_STORAGE_DIR = os.path.join("captchas", "full")
CAPTCHA_CROPPED_DIR = os.path.join("captchas", "cropped")

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
        # Fallback for legacy HumanCooldown objects without retune support
        humanizedCooldown.base = new_value
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
    Saves the latest screenshot for the Anti-Captcha API.
    Stores both the original capture and a cropped captcha image for review.
    Returns the file path used for API submission (cropped image).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        os.makedirs(CAPTCHA_STORAGE_DIR, exist_ok=True)
        os.makedirs(CAPTCHA_CROPPED_DIR, exist_ok=True)
    except Exception as e:
        log_event("ERROR", "save_last_captcha_image", f"Failed to prepare captcha directories: {e}")

    api_output_path = os.path.join(CAPTCHA_CROPPED_DIR, os.path.basename(filename))

    img = save_screenshot(api_output_path)
    if img is None:
        return None

    full_snapshot_path = os.path.join(CAPTCHA_STORAGE_DIR, f"captcha_full_{timestamp}.png")
    try:
        img.save(full_snapshot_path)
    except Exception as e:
        log_event("ERROR", "save_last_captcha_image", f"Failed to archive full captcha snapshot: {e}")

    cropped_img = crop_captcha_image(img)

    if cropped_img is None:
        log_event(
            "ERROR",
            "save_last_captcha_image",
            "Failed to derive captcha crop from screenshot."
        )
        return None

    cropped_snapshot_path = os.path.join(CAPTCHA_CROPPED_DIR, f"captcha_{timestamp}.png")
    try:
        cropped_img.save(cropped_snapshot_path)
    except Exception as e:
        log_event("ERROR", "save_last_captcha_image", f"Failed to save cropped captcha snapshot: {e}")
        return None

    try:
        cropped_img.save(api_output_path)
    except Exception as e:
        log_event("ERROR", "save_last_captcha_image", f"Failed to update API captcha image: {e}")
        return None

    log_event(
        "INFO",
        "save_last_captcha_image",
        "Saved latest captcha image (full & cropped).",
        ocr_text=f"Full: {full_snapshot_path}\nCropped: {cropped_snapshot_path}"
    )
    return api_output_path


def solve_full_screenshot():
    """Captures the full screenshot and sends it to the solver without cropping."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        os.makedirs(CAPTCHA_STORAGE_DIR, exist_ok=True)
    except Exception as e:
        log_event("ERROR", "solve_full_screenshot", f"Failed to prepare captcha directory: {e}")

    full_path = os.path.join(CAPTCHA_STORAGE_DIR, f"captcha_manual_full_{timestamp}.png")

    img = save_screenshot(full_path)
    if img is None:
        log_event("ERROR", "solve_full_screenshot", "Screenshot capture returned no data.")
        return None, None

    log_event(
        "INFO",
        "solve_full_screenshot",
        "Captured full screenshot for manual solver request.",
        ocr_text=f"Saved to: {full_path}",
    )

    solved_code = api_solve_captcha(full_path)
    if not solved_code:
        return None, full_path

    return solved_code, full_path


def detect_captcha_region(img):
    """Locates the captcha image by anchoring off the detected 'anti-bot' text."""
    try:
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception as e:
        log_event("ERROR", "detect_captcha_region", f"Failed to analyze screenshot text: {e}")
        return None

    if not ocr_data.get("text"):
        return None

    target_indices = []
    for idx, text in enumerate(ocr_data["text"]):
        cleaned = text.strip().lower()
        if not cleaned:
            continue
        if "anti-bot" in cleaned:
            target_indices.append(idx)

    if not target_indices:
        return None

    # Use the widest detection of "anti-bot" as the anchor to reduce OCR noise.
    anchor_idx = max(target_indices, key=lambda i: ocr_data["width"][i])
    anchor_left = ocr_data["left"][anchor_idx]
    anchor_top = ocr_data["top"][anchor_idx]
    anchor_width = ocr_data["width"][anchor_idx]
    anchor_height = ocr_data["height"][anchor_idx]

    anchor_bottom = anchor_top + anchor_height
    anchor_center_x = anchor_left + anchor_width / 2

    crop_top = anchor_bottom + CAPTCHA_ANCHOR_VERTICAL_OFFSET
    crop_left = anchor_center_x - (CAPTCHA_ANCHOR_CROP_WIDTH / 2) + CAPTCHA_ANCHOR_HORIZONTAL_ADJUST
    crop_right = crop_left + CAPTCHA_ANCHOR_CROP_WIDTH
    crop_bottom = crop_top + CAPTCHA_ANCHOR_CROP_HEIGHT

    unclamped_box = (crop_left, crop_top, crop_right, crop_bottom)
    crop_box = clamp_crop_box(unclamped_box, img.size)

    if crop_box is None:
        log_event(
            "WARNING",
            "detect_captcha_region",
            "Calculated crop box is outside image bounds.",
            ocr_text=(
                f"Anchor: left={anchor_left}, top={anchor_top}, width={anchor_width}, height={anchor_height}\n"
                f"Unclamped crop: {unclamped_box}"
            )
        )
        return None

    log_event(
        "INFO",
        "detect_captcha_region",
        "Derived captcha crop from 'anti-bot' anchor.",
        ocr_text=(
            f"Anchor: left={anchor_left}, top={anchor_top}, width={anchor_width}, height={anchor_height}\n"
            f"Crop start: left={crop_box[0]}, top={crop_box[1]}"
        )
    )

    return crop_box


def clamp_crop_box(box, img_size):
    """Clamps a crop box to the bounds of the given image size."""
    left, top, right, bottom = box
    width, height = img_size

    left = max(0, min(int(left), width))
    top = max(0, min(int(top), height))
    right = max(left, min(int(right), width))
    bottom = max(top, min(int(bottom), height))

    if left == right or top == bottom:
        return None

    return (left, top, right, bottom)


def crop_captcha_image(img):
    """Crops the captcha image using detected bounds; failure halts downstream flow."""
    try:
        detected_box = detect_captcha_region(img)

        if not detected_box:
            log_event("ERROR", "crop_captcha_image", "Captcha bounds detection failed. No crop available.")
            return None

        log_event(
            "INFO",
            "crop_captcha_image",
            "Detected captcha bounds via OCR search.",
            ocr_text=str(detected_box)
        )
        return img.crop(detected_box)

    except Exception as e:
        log_event("ERROR", "crop_captcha_image", f"Failed to crop captcha image: {e}")
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
    """Sends the CAPTCHA image path to the Anti-Captcha service for solving."""
    if solver is None:
        log_event("ERROR", "api_solve_captcha", "Solver not initialized. Cannot use API.")
        return None

    log_event("API_CALL", "api_solve_captcha", f"Sending image path {img_path} to Anti-Captcha service...")

    result = solver.solve_and_return_solution(img_path)
    captcha_text = result

    if captcha_text != 0:
        log_event("API_CALL", "api_solve_captcha", f"Received solution: **{captcha_text}**")
    else:
        log_event("ERROR", "api_solve_captcha", "TASK FINISHED WITRH ERROR")
        return None

    return captcha_text


def verify_solver_response_timing(code):
    """
    Sends /verify code, then uses the complex 35s/10s timing (solver path).
    Returns True (success and resume) or False (stop script).
    """

    log_event("COMMAND", "verify_solver_response_timing", f"Inputting solver result: /verify {code}")
    clear_input_line()
    time.sleep(0.5)
    keyboard_input("/verify")
    time.sleep(0.1)
    keyboard_press(keyboard.Key.enter)
    keyboard_input(code)
    time.sleep(0.1)
    keyboard_press(keyboard.Key.enter)

    # --- Check 1: Wait 35 seconds ---
    log_event("INFO", "verify_solver_response_timing", "Waiting 35s for verification result.")
    time.sleep(35.0)

    for attempt in range(1, 3):
        img = save_screenshot("verification_check.png")
        raw_text = ocr_screenshot(img)
        raw_text_lower = raw_text.lower()
        log_event("VERIFY", "verify_solver_response_timing", f"Screen check {attempt} after wait.", raw_text)

        if "may now continue" in raw_text_lower:
            log_event("SUCCESS", "verify_solver_response_timing", "Solver code accepted. Resuming loop.")
            return True

        elif "incorrect" in raw_text_lower:
            log_event("FAILURE", "verify_solver_response_timing", "Solver code was incorrect.")
            return False

        if attempt == 1:
            log_event("INFO", "verify_solver_response_timing", "Result unclear. Waiting 10s more.")
            time.sleep(10.0)
        else:
            log_event("CRITICAL", "verify_solver_response_timing",
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
                stop_script("Failed to prepare captcha crop.", details="OCR-based detection did not produce bounds.", exit_program=True)
                return

            solved_code = api_solve_captcha(last_img_path)

            if not solved_code:
                stop_script("Anti-Captcha solver failed.", details="Solver call failed or returned no code.", exit_program=True)
                return

            if not verify_solver_response_timing(solved_code):
                stop_script("Solver result verification failed.", details=f"Solver Code: {solved_code}", exit_program=True)
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
                    f"Captcha solved via Anti-Captcha ({solved_code}). Waiting {delay:.2f}s before resuming. "
                    f"Anti-bot detection disabled for {disable_minutes:.1f} minutes."
                ),
            )

            time.sleep(delay)
            captcha_found.clear()

        else:
            if captcha_found.is_set():
                captcha_found.clear()

    log_event("INFO", "check_loop", "Thread stopped.")

def start_script():
    """Initializes and starts both fish and check threads from the GUI."""
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

    print("\n--- Starting Bot ---")

    check_thread = threading.Thread(target=check_loop, daemon=True)
    check_thread.start()

    time.sleep(0.5)

    fish_thread = threading.Thread(target=fish_loop, daemon=True)
    fish_thread.start()

    print("Bot is running. Ensure Discord is the active window.")


class BotControlGUI(tk.Tk):
    """Graphical control panel for managing the fishing bot."""

    def __init__(self):
        super().__init__()

        self.title("Discord Fishing Bot Control Panel")
        self.geometry("460x360")
        self.resizable(False, False)
        self.configure(bg="#1f2933")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#1f2933")
        style.configure("Card.TFrame", background="#27313d")
        style.configure("Card.TLabelframe", background="#27313d", foreground="#f5f7fa")
        style.configure("Card.TLabelframe.Label", background="#27313d", foreground="#f5f7fa", font=("Segoe UI", 11, "bold"))
        style.configure("Title.TLabel", background="#1f2933", foreground="#f5f7fa", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background="#1f2933", foreground="#9aa5b1", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#27313d", foreground="#f5f7fa", font=("Segoe UI", 12, "bold"))
        style.configure("Detail.TLabel", background="#27313d", foreground="#d9e2ec", font=("Segoe UI", 10))
        style.configure(
            "Control.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(12, 6),
            background="#334155",
            foreground="#f5f7fa",
        )
        style.map("Control.TButton",
                  background=[('active', '#3e4c59'), ('!active', '#334155')],
                  foreground=[('disabled', '#7b8794'), ('!disabled', '#f5f7fa')])
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=(12, 6), background="#1d4ed8", foreground="#f5f7fa")
        style.map("Accent.TButton",
                  background=[('active', '#2563eb'), ('!active', '#1d4ed8')],
                  foreground=[('disabled', '#9aa5b1'), ('!disabled', '#f5f7fa')])

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main_frame, text="Discord Fishing Bot", style="Title.TLabel")
        header.pack(anchor=tk.W)

        subtitle = ttk.Label(
            main_frame,
            text="Control the automation with intuitive buttons and live status updates.",
            style="Subtitle.TLabel",
            wraplength=400,
        )
        subtitle.pack(anchor=tk.W, pady=(0, 15))

        status_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=15)
        status_frame.pack(fill=tk.X, pady=(0, 15))

        self.status_var = tk.StringVar(value="Status: Idle")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(anchor=tk.W)

        self.detail_var = tk.StringVar(value=f"Base cooldown: {BASE_SLEEP:.2f}s")
        detail_label = ttk.Label(status_frame, textvariable=self.detail_var, style="Detail.TLabel")
        detail_label.pack(anchor=tk.W, pady=(5, 0))

        cooldown_control = ttk.Frame(status_frame, style="Card.TFrame")
        cooldown_control.pack(fill=tk.X, pady=(10, 0))

        cooldown_label = ttk.Label(
            cooldown_control,
            text="Adjust base cooldown (seconds):",
            style="Detail.TLabel",
        )
        cooldown_label.grid(row=0, column=0, sticky="w")

        self.cooldown_var = tk.StringVar(value=f"{BASE_SLEEP:.2f}")
        self.cooldown_entry = ttk.Entry(
            cooldown_control,
            textvariable=self.cooldown_var,
            font=("Segoe UI", 11),
        )
        self.cooldown_entry.grid(row=0, column=1, padx=10, sticky="ew")
        self.cooldown_entry.bind("<Return>", lambda _event: self.apply_cooldown())

        self.cooldown_apply = ttk.Button(
            cooldown_control,
            text="Apply",
            style="Control.TButton",
            command=self.apply_cooldown,
            width=10,
        )
        self.cooldown_apply.grid(row=0, column=2, padx=5, sticky="ew")

        cooldown_control.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(main_frame, padding=(0, 5))
        button_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(
            button_frame,
            text="Start",
            style="Accent.TButton",
            command=self.handle_start,
            width=12,
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.pause_button = ttk.Button(
            button_frame,
            text="Pause",
            style="Control.TButton",
            command=self.toggle_pause,
            width=12,
        )
        self.pause_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            style="Control.TButton",
            command=self.handle_stop,
            width=12,
        )
        self.stop_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        button_frame.columnconfigure((0, 1, 2), weight=1)

        manual_frame = ttk.Frame(main_frame, padding=(0, 0))
        manual_frame.pack(fill=tk.X)

        self.solve_button = ttk.Button(
            manual_frame,
            text="Solve Current Screenshot",
            style="Control.TButton",
            command=self.send_full_screenshot_to_solver,
        )
        self.solve_button.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.start_button.state(["!disabled"])
        self.pause_button.state(["disabled"])
        self.stop_button.state(["disabled"])

        self.after(250, self.poll_state)

    def handle_start(self):
        if solver is None:
            messagebox.showerror("Solver not initialized", "Anti-Captcha solver is not initialized. Check your API key.")
            return

        start_script()
        self.poll_state(force=True)

    def handle_stop(self):
        if running:
            stop_script("Script stopped via GUI.")
        else:
            print("Script is already stopped.")
        self.poll_state(force=True)

    def toggle_pause(self):
        if not running:
            messagebox.showinfo("Not running", "Start the bot before attempting to pause it.")
            return

        if paused.is_set():
            resume_script()
        else:
            pause_script()
        self.poll_state(force=True)

    def apply_cooldown(self):
        value = self.cooldown_var.get()
        if update_base_cooldown(value):
            messagebox.showinfo("Cooldown Updated", f"Base cooldown set to {BASE_SLEEP:.2f} seconds.")
            self.detail_var.set(f"Base cooldown: {BASE_SLEEP:.2f}s")
        else:
            messagebox.showerror("Invalid Value", "Please enter a numeric value greater than zero.")
            self.cooldown_var.set(f"{BASE_SLEEP:.2f}")

    def send_full_screenshot_to_solver(self):
        if solver is None:
            messagebox.showerror("Solver not initialized", "Anti-Captcha solver is not initialized. Check your API key.")
            return

        solved_code, image_path = solve_full_screenshot()

        if image_path is None:
            messagebox.showerror(
                "Screenshot Failed",
                "Unable to capture the screen region. Review the logs for more information.",
            )
            return

        if not solved_code:
            messagebox.showerror(
                "Solver Error",
                (
                    "The solver did not return a valid code.\n"
                    f"Screenshot saved at:\n{image_path}"
                ),
            )
            return

        messagebox.showinfo(
            "Solver Response",
            (
                "The solver returned the following code:\n"
                f"{solved_code}\n\n"
                f"Screenshot saved at:\n{image_path}"
            ),
        )

    def poll_state(self, force=False):
        state_text = "Idle"
        if running:
            state_text = "Paused" if paused.is_set() else "Running"
        self.status_var.set(f"Status: {state_text}")

        if not self.cooldown_entry.focus_get() == self.cooldown_entry or force:
            self.cooldown_var.set(f"{BASE_SLEEP:.2f}")

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

if __name__ == "__main__":
    # Initialize the unique log file name on script start
    current_date = datetime.now().strftime('%Y%m%d')
    i = 0
    # Ensure the log file is unique for today's run
    while os.path.exists(f"{current_date}_bot_log_{i}.txt"):
        i += 1
    LOG_FILE = f"{current_date}_bot_log_{i}.txt"

    try:
        solver = imagecaptcha()
        solver.set_verbose(1)
        solver.set_key("ccbfc9559c1b6f36675f0bf5247299ab")
        solver.set_case(1)
        solver.set_minLength(4)
        solver.set_maxLength(8)
        solver.set_comment("[Case Sensitive] image captcha or the code following after Code: )")
        print("Captcha Solver initialized.")
    except Exception as e:
        print(f"!!! CRITICAL: Failed to initialize captcha solver: {e} !!!")
        solver = None

    print("---------------------------------------")
    print("Discord Fishing Bot (GUI Control)")
    print(f"Log File: {LOG_FILE}")
    print("Use the GUI buttons to control the bot.")
    print("---------------------------------------")

    gui = BotControlGUI()
    gui.mainloop()