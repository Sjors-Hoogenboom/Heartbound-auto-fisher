import os
import time
from typing import Tuple, Optional
import re
import pyautogui
import pytesseract
from PIL import Image as PILImage, ImageOps, ImageFilter
import cv2
import numpy as np
from PIL.Image import Resampling

MESSAGE = "/fish"

# Amount of time before each snapshot
POLL_INTERVAL = 1.0

# OCR triggers (lowercased)
TRIGGERS = ("you caught", "nothing seems")
# This text triggers the checking for❗
BANG_TEXT_HINTS = "quick! click the"

# Cooldowns
FISH_COOLDOWN = 8.0

# Tesseract
PYTESSERACT_CMD = r"C:\Users\Sjors\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# OCR tuning
BINARIZE_THRESHOLD = 160
PSM = 6
UPSCALE = 2.0
TAIL_HEIGHT_PX = 220

# Extend hunt if blue is visible
HUNT_EXTEND_IF_BLUE_S = 0.8

# Text to continue looking for ❗
BANG_RETRY_HINTS = ("interaction failed", "too late")

BANG_TEMPLATE = "bang_template.png"     # Image to look for
FISH_PRIME_DELAY = 2.0          # Cooldown before checking for ❗
BANG_SEARCH_WINDOW = 9.0        # How long to look for❗
BANG_SEARCH_INTERVAL = 0.02     # Checks ~50 times per second for the ❗

RED_R_MIN = 210
RED_G_MAX = 80
RED_B_MAX = 80
RED_DOMINANCE = 70

# Requires same spot for N frames to register
BANG_DEBOUNCE_FRAMES = 1
# Determines how big a spotted area is
BANG_DEBOUNCE_TOLERANCE = 12

START_WITH_FISH = True
# Prints the Raw and Bin OCR
DEBUG_OCR = True

# Blue button detection (HSV)
BLUE_HSV_LOW  = (100, 90, 120)
BLUE_HSV_HIGH = (135, 255, 255)

# Only check plausible blue spots
MIN_BLUE_AREA_RATIO = 0.004   # at least 0.4% of buttons_region.png
MAX_BLUE_TILES      = 8       # Only check the largest number of blobs

# Red score we require inside the button's center (7x7 patch)
RED_SCORE_THRESHOLD = 7


def _blue_present_quick(buttons_region: Tuple[int,int,int,int]) -> bool:
    left, top, w, h = buttons_region
    if w <= 0 or h <= 0:
        return False
    shot  = pyautogui.screenshot(region=buttons_region)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(BLUE_HSV_LOW, dtype=np.uint8),
        np.array(BLUE_HSV_HIGH, dtype=np.uint8),
    )
    blue_ratio = float((mask > 0).sum()) / float(max(1, w*h))
    return blue_ratio > 0.01   # ~1% of a region looks blue → likely buttons showing

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

def _preprocess_binarized(img: PILImage.Image) -> PILImage.Image:
    if UPSCALE and UPSCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * UPSCALE), int(h * UPSCALE)), resample=Resampling.LANCZOS)
    gray_img = img.convert("L")
    gray_img = ImageOps.autocontrast(gray_img, cutoff=1)
    binary_img = gray_img.point(lambda p: 255 if p > BINARIZE_THRESHOLD else 0)
    binary_img = binary_img.filter(ImageFilter.SHARPEN)
    return binary_img

def _preprocess_raw(img: PILImage.Image) -> PILImage.Image:
    if UPSCALE and UPSCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * UPSCALE), int(h * UPSCALE)), resample=Resampling.LANCZOS)
    return img

def read_chat_text(region: Tuple[int, int, int, int]) -> Tuple[str, str, PILImage.Image, PILImage.Image]:
    raw = pyautogui.screenshot(region=region)
    raw_up = _preprocess_raw(raw.copy())
    binimg = _preprocess_binarized(raw.copy())
    cfg = f"--oem 3 --psm {PSM}"
    raw_txt = pytesseract.image_to_string(raw_up, config=cfg).lower()
    bin_txt = pytesseract.image_to_string(binimg, config=cfg).lower()
    return raw_txt, bin_txt, raw_up, binimg

def calibrate_region_with_mouse(prompt_name: str) -> Tuple[int,int,int,int]:
    print(f"\nCalibrating {prompt_name} region:")
    print("1) Hover your mouse over the TOP-LEFT of the area")
    for s in [3,2,1]:
        print(f"   capturing in {s}…"); time.sleep(1)
    tl = pyautogui.position()
    print(f"   top-left = ({tl.x}, {tl.y})")

    print("2) Hover your mouse over the BOTTOM-RIGHT of the area")
    for s in [3,2,1]:
        print(f"   capturing in {s}…"); time.sleep(1)
    br = pyautogui.position()
    print(f"   bottom-right = ({br.x}, {br.y})")

    left, top = min(tl.x, br.x), min(tl.y, br.y)
    width, height = abs(br.x - tl.x), abs(br.y - tl.y)
    region = (left, top, width, height)
    print(f"→ Calibrated {prompt_name} = {region}")

    test = pyautogui.screenshot(region=region)
    test.save(f"{prompt_name.replace(' ','_')}.png")
    print(f"Saved {prompt_name.replace(' ','_')}.png — check the crop.")
    return region

def bottom_tail(region: Tuple[int,int,int,int], tail_h: int) -> Tuple[int,int,int,int]:
    left, top, width, height = region
    tail_h = min(tail_h, height)
    tail_top = top + (height - tail_h)
    return left, tail_top, width, tail_h

def count_triggers(raw_txt: str, bin_txt: str) -> int:
    total = 0
    for t in TRIGGERS:
        total += max(raw_txt.count(t), bin_txt.count(t))
    return total

_whitespace_re = re.compile(r"\s+")
_nonword_re = re.compile(r"[^a-z0-9\s]+")
def _normalize_line(s: str) -> str:
    s = s.lower()
    s = _nonword_re.sub("", s)
    s = _whitespace_re.sub(" ", s).strip()
    return s

def _last_trigger_line_key(raw_txt: str, bin_txt: str) -> Optional[str]:
    def pick_line(txt: str) -> Optional[str]:
        lines = [ln for ln in txt.splitlines() if any(t in ln for t in TRIGGERS)]
        if not lines:
            return None
        return max(lines, key=len)
    cand_raw = pick_line(raw_txt)
    cand_bin = pick_line(bin_txt)
    best = cand_raw if (cand_raw and (not cand_bin or len(cand_raw) >= len(cand_bin))) else cand_bin
    return _normalize_line(best) if best else None

# ---------- ❗ helpers ----------
def _safely_locate(template_path, region, confidence):
    try:
        return pyautogui.locateOnScreen(template_path, region=region, confidence=confidence)
    except Exception:
        return None

def _red_score_at(x, y, patch=7) -> int:
    """Count 'very red' pixels in a small square around (x,y)."""
    half = patch // 2
    left = max(0, x - half)
    top  = max(0, y - half)
    shot = pyautogui.screenshot(region=(left, top, patch, patch))
    px = shot.load()
    score = 0
    for yy in range(patch):
        for xx in range(patch):
            r, g, b = px[xx, yy][:3]
            if (r >= RED_R_MIN and g <= RED_G_MAX and b <= RED_B_MAX and
                (r - max(g, b)) >= RED_DOMINANCE):
                score += 1
    return score

def _find_blue_tiles(frame_bgr: np.ndarray, region_area: int):
    """Return list of bounding rects (x,y,w,h) for blue button tiles, largest first."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    # clean small speckles / close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], mask

    tiles = []
    min_area = max(40, int(MIN_BLUE_AREA_RATIO * region_area))
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        # heuristics: big enough, roughly rectangular tile (w>>h or ~square),
        # ignore ultra-thin ribbons
        if area >= min_area and w >= 30 and h >= 25:
            tiles.append((x,y,w,h))
    # largest first
    tiles.sort(key=lambda r: r[2]*r[3], reverse=True)
    return tiles[:MAX_BLUE_TILES], mask


def watch_and_click_bang(buttons_region: Tuple[int,int,int,int], window_s: float) -> bool:
    """
    Every frame:
      1) fresh screenshot of buttons_region
      2) find blue tiles
      3) score each tile's center for 'redness'
      4) debounce the best center across frames, then click
    """
    left, top, w, h = buttons_region
    region_area = max(1, w*h)

    deadline = time.time() + window_s
    last_center = None
    stable_frames = 0
    attempts = 0

    while time.time() < deadline:
        shot  = pyautogui.screenshot(region=buttons_region)
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

        tiles, blue_mask = _find_blue_tiles(frame, region_area)
        attempts += 1

        if not tiles:
            # nothing blue this frame
            last_center = None
            stable_frames = 0
            time.sleep(BANG_SEARCH_INTERVAL)
            continue

        # Pick the tile whose center has the strongest red score
        best = None  # (score, cx, cy, rect)
        for (x,y,ww,hh) in tiles:
            cx = left + x + ww//2
            cy = top  + y + hh//2
            score = _red_score_at(cx, cy, patch=7)
            if best is None or score > best[0]:
                best = (score, cx, cy, (x,y,ww,hh))

        score, cx, cy, rect = best

        # Require a little red at the center
        if score >= RED_SCORE_THRESHOLD:
            # Debounce: same spot across frames
            if last_center and abs(cx - last_center[0]) <= BANG_DEBOUNCE_TOLERANCE and abs(cy - last_center[1]) <= BANG_DEBOUNCE_TOLERANCE:
                stable_frames += 1
            else:
                stable_frames = 1
                last_center = (cx, cy)

            if stable_frames >= BANG_DEBOUNCE_FRAMES:
                pyautogui.click(cx, cy)
                print(f"Clicked ❗ (blue→red method) at ({cx},{cy}) score={score} after {attempts} frames")
                return True
        else:
            # center not red enough yet; keep hunting
            last_center = None
            stable_frames = 0

        time.sleep(BANG_SEARCH_INTERVAL)

    print("❗ not found within window")
    return False

_BANG_TPL = None  # cv2 grayscale template (loaded in main)

def _load_bang_template() -> Optional[np.ndarray]:
    if not os.path.exists(BANG_TEMPLATE):
        print(f"[WARN] Missing template '{BANG_TEMPLATE}'. Make a tight crop of the red ❗ at current zoom.")
        return None
    tpl = cv2.imread(BANG_TEMPLATE, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        print(f"[WARN] Could not read '{BANG_TEMPLATE}' (corrupt or unsupported).")
    return tpl

def main():
    _ensure_tesseract()
    global _BANG_TPL
    _BANG_TPL = _load_bang_template()

    chat_full = calibrate_region_with_mouse("chat region")
    chat_region = bottom_tail(chat_full, TAIL_HEIGHT_PX)

    buttons_region = calibrate_region_with_mouse("buttons region")

    print("You have 3 seconds to focus the game/chat window…")
    time.sleep(3)

    # hunt window
    hunt_start_ts: Optional[float] = None
    hunt_until_ts: Optional[float] = None

    if START_WITH_FISH:
        send_message(MESSAGE)
        now = time.time()
        hunt_start_ts = now + FISH_PRIME_DELAY
        hunt_until_ts = hunt_start_ts + BANG_SEARCH_WINDOW
        print(f"[bang] armed (prime {FISH_PRIME_DELAY}s, window {BANG_SEARCH_WINDOW}s)")
        print(f"Cooling down {FISH_COOLDOWN}s after initial /fish…")
        time.sleep(FISH_COOLDOWN)

    last_trigger_count = 0
    last_trigger_key: Optional[str] = None
    i = 0

    try:
        while True:
            now = time.time()

            # --- OCR the chat tail ---
            raw_txt, bin_txt, raw_img, bin_img = read_chat_text(chat_region)

            if DEBUG_OCR:
                print("RAW OCR:", repr(raw_txt.strip()[:220]))
                print("BIN OCR:", repr(bin_txt.strip()[:220]))

            # 1) open/extend hunt on normal hint
            if any(h in raw_txt or h in bin_txt for h in BANG_TEXT_HINTS):
                hunt_start_ts = now
                hunt_until_ts = now + BANG_SEARCH_WINDOW
                print(f"Hint detected → hunting ❗ for {BANG_SEARCH_WINDOW}s")

            # 2) also open/extend hunt on retry text (your bug case)
            if any(h in raw_txt or h in bin_txt for h in BANG_RETRY_HINTS):
                hunt_start_ts = now
                hunt_until_ts = now + BANG_SEARCH_WINDOW
                print(f"Retry text detected → hunting ❗ for {BANG_SEARCH_WINDOW}s")

            # 3) if we can *see* blue buttons, keep the hunt alive even if OCR missed text
            if hunt_start_ts is not None and (hunt_until_ts is None or now <= hunt_until_ts):
                if _blue_present_quick(buttons_region):
                    # push the window forward a bit so it stays open while blue is visible
                    hunt_start_ts = min(hunt_start_ts or now, now)
                    hunt_until_ts = max(hunt_until_ts or now, now + HUNT_EXTEND_IF_BLUE_S)

            # 4) while the hunt window is open, do short high-FPS slices
            if hunt_start_ts is not None and hunt_until_ts is not None and hunt_start_ts <= now <= hunt_until_ts:
                print("[bang] scanning slice…")
                watch_and_click_bang(buttons_region, 0.4)

            # 5) send /fish on result lines
            occurrences = count_triggers(raw_txt, bin_txt)
            key = _last_trigger_line_key(raw_txt, bin_txt)

            trigger_now = False
            dbg_reason = "no change"
            if key and key != last_trigger_key:
                trigger_now = True
                dbg_reason = f"new line key (prev={last_trigger_key!r}, now={key!r})"
            elif occurrences > last_trigger_count:
                trigger_now = True
                dbg_reason = f"count increased ({last_trigger_count} -> {occurrences})"

            if trigger_now:
                print(f"Detected RESULT via {dbg_reason} → sending {MESSAGE!r}")
                send_message(MESSAGE)

                # re-arm next hunt relative to this cast (time-based, robust even if hint OCR fails)
                now = time.time()
                hunt_start_ts = now + FISH_PRIME_DELAY
                hunt_until_ts = hunt_start_ts + BANG_SEARCH_WINDOW
                print(f"[bang] armed (prime {FISH_PRIME_DELAY}s, window {BANG_SEARCH_WINDOW}s)")

                print(f"Cooling down {FISH_COOLDOWN}s before next OCR check…")
                time.sleep(FISH_COOLDOWN)

            last_trigger_key = key or last_trigger_key
            last_trigger_count = occurrences

            i += 1
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
