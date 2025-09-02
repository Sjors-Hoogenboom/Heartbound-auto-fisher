import os
import time
from typing import Tuple

import pyautogui
import pytesseract
from PIL import Image as PILImage, ImageOps, ImageFilter

MESSAGE = "/fish"
POLL_INTERVAL = 5.0

# Messages the script will look for
TRIGGERS = ("you caught", "nothing seems")

# Cooldown before it takes a new snapshot
FISH_COOLDOWN = 15.0

# Tesseract path should be something like r"C:\Users\username\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
PYTESSERACT_CMD =

# OCR tuning
BINARIZE_THRESHOLD = 160
PSM = 6
UPSCALE = 2.0

TAIL_HEIGHT_PX = 220

START_WITH_FISH = True
DEBUG_OCR = True
SAVE_DEBUG_SHOTS = False

def _ensure_tesseract():
    if PYTESSERACT_CMD and os.path.exists(PYTESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_CMD
        return
    from shutil import which
    found = which("tesseract")
    if not found:
        raise RuntimeError("Tesseract not found. Set PYTESSERACT_CMD to tesseract.exe.")
    pytesseract.pytesseract.tesseract_cmd = found

def send_message(msg: str):
    pyautogui.write(msg)
    pyautogui.press('enter')
    pyautogui.press('enter')
    print(f"Sent '{msg}'")

def preprocess_binarized(img: PILImage.Image) -> PILImage.Image:
    if UPSCALE and UPSCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * UPSCALE), int(h * UPSCALE)), resample=PILImage.LANCZOS)
    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.point(lambda p: 255 if p > BINARIZE_THRESHOLD else 0)
    g = g.filter(ImageFilter.SHARPEN)
    return g

def preprocess_raw(img: PILImage.Image) -> PILImage.Image:
    if UPSCALE and UPSCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * UPSCALE), int(h * UPSCALE)), resample=PILImage.LANCZOS)
    return img

def read_chat_text(region: Tuple[int, int, int, int]) -> Tuple[str, str, PILImage.Image, PILImage.Image]:
    raw = pyautogui.screenshot(region=region)
    raw_up = preprocess_raw(raw.copy())
    binimg = preprocess_binarized(raw.copy())
    cfg = f"--oem 3 --psm {PSM}"
    raw_txt = pytesseract.image_to_string(raw_up, config=cfg).lower()
    bin_txt = pytesseract.image_to_string(binimg, config=cfg).lower()
    return raw_txt, bin_txt, raw_up, binimg

def calibrate_region_with_mouse() -> Tuple[int,int,int,int]:
    print("\nCalibrating chat region:")
    print("1) Hover your mouse over the TOP-LEFT of the chat")
    for s in [3,2,1]:
        print(f"   capturing in {s}…"); time.sleep(1)
    tl = pyautogui.position()
    print(f"   top-left = ({tl.x}, {tl.y})")

    print("2) Hover your mouse over the BOTTOM-RIGHT of the chat")
    for s in [3,2,1]:
        print(f"   capturing in {s}…"); time.sleep(1)
    br = pyautogui.position()
    print(f"   bottom-right = ({br.x}, {br.y})")

    left, top = min(tl.x, br.x), min(tl.y, br.y)
    width, height = abs(br.x - tl.x), abs(br.y - tl.y)
    region = (left, top, width, height)
    print(f"→ Calibrated CHAT_REGION = {region}")

    test = pyautogui.screenshot(region=region)
    test.save("chat_region.png")
    print("Saved chat_region.png check if it's correct")
    return region

def bottom_tail(region: Tuple[int,int,int,int], tail_h: int) -> Tuple[int,int,int,int]:
    """Return just the bottom tail of the calibrated region."""
    left, top, width, height = region
    tail_h = min(tail_h, height)
    tail_top = top + (height - tail_h)
    return left, tail_top, width, tail_h

def count_triggers(raw_txt: str, bin_txt: str) -> int:
    total = 0
    for t in TRIGGERS:
        total += max(raw_txt.count(t), bin_txt.count(t))
    return total

def main():
    _ensure_tesseract()

    full_region = calibrate_region_with_mouse()
    region = bottom_tail(full_region, TAIL_HEIGHT_PX)

    print("You have 5 seconds to focus the game/chat window…")
    time.sleep(5)

    if START_WITH_FISH:
        send_message(MESSAGE)
        print(f"Cooling down {FISH_COOLDOWN}s after initial /fish…")
        time.sleep(FISH_COOLDOWN)

    last_trigger_count = 0
    i = 0

    try:
        while True:
            print(f"Scanning region {region} every {POLL_INTERVAL:.1f}s. Triggers: {TRIGGERS}")
            raw_txt, bin_txt, raw_img, bin_img = read_chat_text(region)

            if DEBUG_OCR:
                print("RAW OCR:", repr(raw_txt.strip()[:300]))
                print("BIN OCR:", repr(bin_txt.strip()[:300]))

            if SAVE_DEBUG_SHOTS:
                raw_img.save(f"ocr_raw_{i:04d}.png")
                bin_img.save(f"ocr_bin_{i:04d}.png")

            occurrences = count_triggers(raw_txt, bin_txt)

            if occurrences > last_trigger_count:
                print(f"Detected NEW trigger (count {last_trigger_count} -> {occurrences}) → sending {MESSAGE!r}")
                send_message(MESSAGE)
                print(f"Cooling down {FISH_COOLDOWN}s before next OCR check…")
                time.sleep(FISH_COOLDOWN)

            last_trigger_count = occurrences

            i += 1
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
