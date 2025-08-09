import pyautogui
import time
import random

# Config
message = "/fish"
base_delay = 1.7        #base delay
random_range = 1.1      #random delay


def send_message(msg):
    pyautogui.write(msg)
    pyautogui.press('enter')
    pyautogui.press('enter')

print("5 seconds to change window")
time.sleep(5)

try:
    while True:
        send_message(message)
        delay = base_delay + random.uniform(0, random_range)
        print(f"Sent '{message}', waiting {delay:.2f} seconds"
              f"/fish"
              f""
              f"...")
        time.sleep(delay)
except KeyboardInterrupt:
    print("\nStopped.")
