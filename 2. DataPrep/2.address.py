import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Set up Selenium options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Path to your WebDriver
webdriver_service = Service('C:\Program Files\chromedriver-win64\chromedriver.exe')  # Change this to your path
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

df_filtered = pd.read_csv("updated_file.csv")

# Define error handling and restarting variables
error_count = 0
error_threshold = 10  # Number of errors after which to trigger a restart
batch_size = 100
output_file = 'address_out.csv'
changes_made = False

# Track repeated failures across batches
repeated_failures = 0     # Tracks how many times the same errors have repeated

# Function to get the address of a company from Google Search using Selenium
def get_company_address(company_name):
    query = f"{company_name} address"
    url = f"https://www.google.com/search?q={query}"
    
    try:
        driver.get(url)  # Open the URL
        
        # Use WebDriverWait to wait for the address element to be present
        wait = WebDriverWait(driver, 1)  # Wait for up to 10 seconds

        address_div = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'sXLaOe'))
        )
        address = address_div.text.strip()
        print(f"Found address for {company_name}: {address}")  # Add this to verify the result
        return address

    except Exception as e:
        print(f"Error retrieving address for {company_name}")
        return np.nan  # If the element is not found, return NaN

# Check if 'last_processed.txt' exists to resume processing
try:
    with open('last_processed.txt', 'r') as f:
        last_processed = int(f.readline().strip())  # Read the last processed index
        print(f'starting from index: {last_processed}')
except FileNotFoundError:
    last_processed = -1  # If no log file, start from the beginning

# Check if 'repeated_failures.txt' exists to resume repeated failure count
try:
    with open('repeated_failures.txt', 'r') as f:
        repeated_failures = int(f.readline().strip())  # Read the repeated failures count
except FileNotFoundError:
    repeated_failures = 0  # If no log file, start with zero repeated failures

# Resume from last processed index
try:
    for idx, row in df_filtered.iterrows():
        if idx <= last_processed:
            continue  # Skip rows that have already been processed

        if pd.isna(df_filtered.at[idx, 'address']):
            address = get_company_address(row['company'])
            if pd.isna(address):
                error_count += 1
                print(f"Error count: {error_count}")
            else:
                df_filtered.at[idx, 'address'] = address
                error_count = 0  # Reset the error count on successful retrieval
                repeated_failures = 0  # Reset repeated failures on success
                changes_made = True

                if repeated_failures > 0:
                    # Immediately update repeated_failures.txt when a successful address is found
                    with open('repeated_failures.txt', 'w') as f:
                        f.write(str(repeated_failures))
                    print("Repeated batch failures reset back to 0")

        # If error count exceeds threshold, handle repeated failures
        if error_count >= error_threshold:
            repeated_failures += 1  # Increment repeated failures

            if repeated_failures > 1:
                print(f"Batch of 10 failures detected twice. Restarting without moving back.")
                error_count = 0  # Reset error count
                repeated_failures = 0  # Reset repeated failure count
                # Update repeated_failures.txt when resetting after second batch of failures
                with open('repeated_failures.txt', 'w') as f:
                    f.write(str(repeated_failures))  # Reset repeated_failures count
                print('Repeated batch failure happened twice, skipping 10 rows.')

                continue  # Move on without resetting the index

            else:
                # First batch of failures, restart at index - 10 (but not less than 0)
                restart_index = max(0, idx - 10)
                print(f"Error threshold reached. Restarting at index {restart_index}.")
                if changes_made:
                    df_filtered.to_csv(output_file, index=False)
                    # Save last processed index
                    with open('last_processed.txt', 'w') as f:
                        f.write(f"{restart_index}\n")  # Save the restart index
                    # Save repeated failures count
                    with open('repeated_failures.txt', 'w') as f:
                        f.write(str(repeated_failures))  # Save repeated failures count
                driver.quit()
                os._exit(1)  # Exit the script, allowing for a restart mechanism

        # Save progress in batches
        if (idx + 1) % batch_size == 0 and changes_made:
            df_filtered.to_csv(output_file, index=False)
            # Save the progress, including repeated failures count
            with open('last_processed.txt', 'w') as f:
                f.write(f"{idx}\n")  # Log the last processed index
            with open('repeated_failures.txt', 'w') as f:
                f.write(str(repeated_failures))  # Save repeated failures count
            print(f"Progress saved after processing {idx + 1} rows.")
            changes_made = False  # Reset the flag after saving

    # Final save after completing all rows
    df_filtered.to_csv(output_file, index=False)

finally:
    driver.quit()  # Ensure the driver is closed
