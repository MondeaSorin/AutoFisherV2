# AutoFisherV2

This project automates Discord fishing commands while reacting to captcha challenges using the 2Captcha service.

## Prerequisites

- Python 3.10+
- Tesseract OCR installed locally (update `pytesseract.pytesseract.tesseract_cmd` if the path differs)
- A valid [2Captcha](https://2captcha.com/) API key

Install required Python packages:

```bash
pip install -r requirements.txt
```

> If a `requirements.txt` file is not available in your working copy, install the packages referenced at the top of `AutoFisher.py` manually.

## Configuration

1. Set the `APIKEY_2CAPTCHA` environment variable with your 2Captcha key. If the variable is missing the script will fall back to the key defined inside `AutoFisher.py`.
2. Ensure the `CAPTURE_AREA` rectangle in `AutoFisher.py` matches the region where Discord displays captcha prompts on your screen.
3. Adjust the `pytesseract.pytesseract.tesseract_cmd` path so it points to your local Tesseract installation.

## Running the Bot

1. Start Discord and focus the channel where you want to fish.
2. Run the script:

   ```bash
   python AutoFisher.py
   ```

3. Use the function keys to control the bot:
   - **F8** – start fishing and screen monitoring threads
   - **F9** – stop the bot immediately
   - **F10** – capture a test screenshot and run it through OCR/2Captcha

The refactored captcha flow watches for the text `Anti-bot` and immediately submits the latest screenshot to 2Captcha. After a successful solve, the bot:

- Waits a random 4–10 seconds before resuming `/fish`
- Suppresses new captcha triggers for 10 minutes
- Logs every major step (capture, API call, verification) to a dated log file in the project root

## Troubleshooting

- If the solver fails to initialize, check that your network allows outbound requests to the 2Captcha API and that the API key is correct.
- If OCR verification does not detect `Anti-bot`, tweak the `CAPTURE_AREA` coordinates or confirm the Discord client is running with standard theme/font sizes.
- Review the generated `*_bot_log_*.txt` files for detailed diagnostics when the bot exits unexpectedly.

## Integrating into an Existing Setup

To merge this refactor into an older checkout:

1. Back up your existing `AutoFisher.py`.
2. Copy the updated `AutoFisher.py` into your project root.
3. Replace or merge the package docstrings in `src/__init__.py`, `src/Utils/__init__.py`, and `src/Debug/__init__.py` so they match this repository if you use those packages.
4. Re-run `python -m compileall AutoFisher.py` (or your preferred tests) to confirm syntax correctness.

These steps ensure the bot immediately relies on 2Captcha when the `Anti-bot` message appears and honors the new cooldown safeguards.
