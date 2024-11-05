import subprocess
import time

while True:
    print("Running scraping script...")
    result = subprocess.run(['python', 'address.py'])
    if result.returncode != 0:
        print("Script crashed or reached error threshold. Restarting in 5 seconds...")
        time.sleep(5)
    else:
        print("Script completed successfully.")
        break
