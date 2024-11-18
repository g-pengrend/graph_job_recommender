import pandas as pd
from geopy.geocoders import Nominatim
import re
import os

# Initialize Nominatim geolocator
geolocator = Nominatim(user_agent="job_recommender")

# Function to extract postal code (Singapore format or 6-digit) from address
def extract_postal_code(address):
    # Try to match the "Singapore XXXXXX" format first
    match = re.search(r"Singapore \d{6}", address)
    if match:
        return match.group(0)
    
    # If "Singapore XXXXXX" not found, check for just a 6-digit number
    match = re.search(r"\d{6}", address)
    if match:
        # Prepend "Singapore" to the 6-digit postal code
        return f"Singapore {match.group(0)}"
    
    return None  # Return None if no valid postal code found

def remove_digits(address):
    return re.sub(r'\d+', '', address).strip()

# Function to geolocate the address
def geocode_location(address):
    try:
        # Try to geolocate using the full address
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        
        # 2. If no location found, try to remove digits and geocode again
        address_no_digits = remove_digits(address)
        location = geolocator.geocode(address_no_digits)
        if location:
            return (location.latitude, location.longitude)
        
        # If no location found, attempt to use just the postal code
        postal_code = extract_postal_code(address)
        if postal_code:
            location = geolocator.geocode(postal_code)
            return (location.latitude, location.longitude) if location else None
        
        return None  # Return None if both attempts fail
    except:
        return None  # Catch any other exceptions and return None

# Apply the function to df2['address'] and create a new column 'lat_long'
def process_geolocation(row):
    lat_long = geocode_location(row['cleaned_address'])
    print(f"Processed: {row['address']} -> {lat_long}")  # Print out the result after processing
    return lat_long

# Function to process the data
def process_data(df, output_file):
    try:
        # Only process rows where 'lat_long' is NaN or missing
        for idx, row in df.iterrows():
            if pd.isna(row['lat_long']):
                df.at[idx, 'lat_long'] = process_geolocation(row)

        # Save the fully processed DataFrame
        df.to_csv(output_file, index=False)
        print("Geolocation process completed and saved to file.")

    except Exception as e:
        print(f"\nProcess interrupted due to: {type(e).__name__}. Saving progress...")
        # Save the progress made before interruption
        df.to_csv('updated_file_with_partial_geolocation.csv', index=False)
        print("Progress saved to 'updated_file_with_partial_geolocation.csv'. Exiting gracefully.")
        exit()

# Main function to handle first-time processing or resuming
def main():
    partial_file = 'updated_file_with_partial_geolocation.csv'
    original_file = '4.1.cleaned_address_out.csv'  # Replace with your original CSV file
    output_file = 'updated_file_with_geolocation.csv'

    # Check if partial progress file exists
    if os.path.exists(partial_file):
        print(f"Resuming from {partial_file}...")
        df = pd.read_csv(partial_file)
    else:
        print(f"Starting fresh from {original_file}...")
        df = pd.read_csv(original_file)
        
        # Initialize the 'lat_long' column with NaN if not already present
        if 'lat_long' not in df.columns:
            df['lat_long'] = None

    # Start or resume processing the data
    process_data(df, output_file)

# Run the main function
if __name__ == "__main__":
    main()