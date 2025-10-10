import mss
from PIL import Image
import pytesseract
from pynput import keyboard
import re
import time
from datetime import datetime
from itertools import product
import threading
import sys
import os
from twocaptcha import TwoCaptcha
from twocaptcha import ApiException
from src.CooldownGenerator.cooldown_humanizer import HumanCooldown

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

# OCR FUZZY CANDIDATE CONFIG
COMMON_ERRORS = {
    'c': ['c', 'e', 'o'], 'e': ['e', 'c'],
    'o': ['o', 'c', '0', 'O'], 'i': ['i', 'l', '1'],
    'l': ['l', 'i', '1'], 's': ['s', '5'],
    'z': ['z', '2'], '0': ['0', 'o', 'O'],
    '1': ['1', 'l', 'i'], '5': ['5', 's', 'S'],
    '2': ['2', 'z', 'Z'], '4': ['4', 'A', 'H'],
}

# =====================================================================
# --- 2. GLOBAL STATE AND THREAD CONTROLS ---
# =====================================================================

running = False
fish_thread = None
check_thread = None
solver = None
captcha_found = threading.Event()
LAST_CANDIDATE_SET = set()
TERMINATION_REASON = "Script manually started/stopped."

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
        f"  Last Candidates: {list(LAST_CANDIDATE_SET)}\n"
        f"----------------------------------------\n"
    )

    try:
        with open(LOG_FILE, 'a') as f:
            f.write(summary)
    except Exception:
        pass

    print(f"\n[LOGGER] Termination reason saved to {LOG_FILE}.")


def stop_script(reason="Unknown Error", details=""):
    """Gracefully stops all loops, logs the reason, and exits the program."""
    global running
    global TERMINATION_REASON

    if not running:
        print("Script is already stopped.")
        return

    TERMINATION_REASON = reason
    print(f"\n--- Stopping Bot ({reason}) ---\n")
    running = False
    captcha_found.set()  # Release any waiting threads

    time.sleep(1)
    log_termination(TERMINATION_REASON, details)
    sys.exit(0)


# =====================================================================
# --- 4. CORE UTILITY FUNCTIONS ---
# =====================================================================

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


def generate_candidates(code):
    """Generates alternative code candidates based on common OCR errors."""
    if not code or len(code) < 4:
        return []

    original_ocr_code = code
    substitution_groups = []

    for char in code:
        char_lower = char.lower()
        group = []

        if char_lower in COMMON_ERRORS:
            for alt in COMMON_ERRORS[char_lower]:
                if char.isupper() and alt.isalpha():
                    group.append(alt.upper())
                else:
                    group.append(alt)

        group.append(char)
        substitution_groups.append(list(set(group)))

    candidate_tuples = list(product(*substitution_groups))
    candidates = ["".join(t) for t in candidate_tuples if len("".join(t)) == len(code)]

    unique_candidates = list(dict.fromkeys(candidates))
    if original_ocr_code in unique_candidates:
        unique_candidates.remove(original_ocr_code)
    unique_candidates.insert(0, original_ocr_code)

    return unique_candidates


# =====================================================================
# --- 5. VERIFICATION LOGIC FUNCTIONS ---
# =====================================================================

def verify_ocr_candidate(code):
    """
    Submits /verify code and checks for result with 10s + 10s logic (OCR Path).
    Returns "may now continue", "incorrect", or "stop".
    """

    log_event("COMMAND", "verify_ocr_candidate", f"Inputting: /verify {code}")
    clear_input_line()
    time.sleep(1)
    keyboard_input("/verify")
    keyboard_press(keyboard.Key.enter)
    time.sleep(1)
    keyboard_input(code)
    keyboard_press(keyboard.Key.enter)

    # --- Check 1: Wait 10 seconds ---
    log_event("INFO", "verify_ocr_candidate", "Waiting 10s for result (Check 1).")
    time.sleep(10.0)

    for attempt in range(1, 3):
        img = save_screenshot("verification_check.png")  # Temp screenshot for OCR check
        raw_text = ocr_screenshot(img)
        raw_text_lower = raw_text.lower()
        log_event("VERIFY", "verify_ocr_candidate", f"Screen check {attempt} after wait.", raw_text)

        if "may now continue" in raw_text_lower:
            log_event("SUCCESS", "verify_ocr_candidate", "Code accepted ('may now continue' found).")
            return "may now continue"

        elif "incorrect" in raw_text_lower:
            log_event("FAILURE", "verify_ocr_candidate", "Code rejected ('incorrect' found).")
            return "incorrect"

        if attempt == 1:
            log_event("INFO", "verify_ocr_candidate", "Result unclear. Waiting 10s more (Check 2).")
            time.sleep(10.0)
        else:
            log_event("CRITICAL", "verify_ocr_candidate", "Verification failed to get clear response after 20 seconds.")
            return "stop"

    return "stop"


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


def solve_captcha(candidates):
    """
    Attempts to solve the captcha using the fuzzy candidates list.
    Returns True if solved and fishing should resume, False if exhausted/critical error.
    """
    log_event("SOLVER", "solve_captcha", f"Starting Fuzzy Solver with {len(candidates)} candidates.")

    # Start from the second candidate (the first one failed the primary check)
    start_index = 1
    if len(candidates) <= 1:
        stop_script("Candidate list was critically short.",
                    details="Original OCR failed and no substitutions were possible.")
        return False

    for i in range(start_index, len(candidates)):
        candidate = candidates[i]
        log_event("SOLVER", "solve_captcha", f"Attempting candidate {i + 1}/{len(candidates)}: **{candidate}**")

        result = verify_ocr_candidate(candidate)

        if result == "may now continue":
            log_event("SUCCESS", "solve_captcha", f"Candidate **{candidate}** accepted.")
            return True
        elif result == "incorrect":
            pass
        elif result == "stop":
            return False

    stop_script("Candidate Exhaustion", details="Fuzzy Candidate list exhausted without finding the correct code.")
    return False


# =====================================================================
# --- 6. MAIN THREAD LOOPS ---
# =====================================================================

def fish_loop():
    """Main loop: continuously inputs /fish command."""
    global running
    global humanizedCooldown

    while running:
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
    global LAST_CANDIDATE_SET

    while running:
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

            if not captcha_found.is_set():
                captcha_found.set()
                log_event("CAPTCHA", "check_loop", "CAPTCHA detected. Pausing fish loop.", raw_text)

            primary_match = re.search(r'Code:\s*([A-Za-z0-9]+)', raw_text)

            if primary_match:
                # --- PATH 1: FUZZY CANDIDATE PATH (Text CAPTCHA) ---
                captcha_code = primary_match.group(1).strip()

                # CACHE CHECK: Prevents re-attempting the solve on the same code.
                if LAST_CANDIDATE_SET and captcha_code in LAST_CANDIDATE_SET:
                    log_event("INFO", "check_loop",
                              f"Detected code **{captcha_code}** is a repeat. Waiting (Cache Hit).")

                    # 🚀 FIX: Clear the flag here to signal the fish_loop to resume.
                    captcha_found.clear()

                    time.sleep(1)
                    continue

                candidates = generate_candidates(captcha_code)
                LAST_CANDIDATE_SET = set(candidates)
                original_ocr_candidate = candidates[0]

                log_event("SOLVER", "check_loop", f"OCR detected code: {captcha_code}. Starting primary verification.")

                result = verify_ocr_candidate(original_ocr_candidate)

                if result == "may now continue":
                    captcha_found.clear()

                elif result == "incorrect":
                    if solve_captcha(candidates):
                        captcha_found.clear()

                elif result == "stop":
                    return

            else:
                # --- PATH 2: API PATH (Image CAPTCHA) ---
                log_event("API_PATH", "check_loop", "No 'Code:' string found. Switching to 2Captcha API.", raw_text)

                last_img_path = save_last_captcha_image()
                solved_code = api_solve_captcha(last_img_path)

                if solved_code:
                    if verify_api_response_timing(solved_code):
                        captcha_found.clear()
                        LAST_CANDIDATE_SET = {solved_code}
                    else:
                        stop_script(f"API result verification failed.", details=f"API Code: {solved_code}")
                        return
                else:
                    stop_script("2Captcha API failed.", details="API call failed or returned no code.")
                    return

        else:
            # --- DEFINITIVE CLEARING LOGIC ---
            if not captcha_found.is_set() and LAST_CANDIDATE_SET:
                log_event("INFO", "check_loop", "Captcha code vanished from screen. Clearing cache.")
                LAST_CANDIDATE_SET = set()

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

    primary_match = re.search(r'Code:\s*([A-Za-z0-9]+)', raw_text)

    if primary_match:
        captcha_code = primary_match.group(1).strip()
        candidates = generate_candidates(captcha_code)

        print("\n==================================")
        print(f"OCR Result: **{captcha_code}**")
        print(f"Fuzzy Candidates: {len(candidates) - 1}")
        print("==================================")

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
    global LAST_CANDIDATE_SET

    if running:
        print("Script is already running.")
        return

    running = True
    captcha_found.clear()
    LAST_CANDIDATE_SET = set()
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

    listen_for_f_keys()