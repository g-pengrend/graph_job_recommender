import pandas as pd
from geopy.geocoders import Nominatim
import re
import os
import time

# Initialize Nominatim geolocator
geolocator = Nominatim(user_agent="job_recommender")

# Function to extract postal code (Singapore format or 6-digit) from address
def extract_postal_code(address):
    # Look for a 6-digit postal code
    match = re.search(r"\d{6}", address)
    if match:
        return match.group(0)  # Return as string to preserve leading zeros
    
    return None  # Return None if no valid postal code found

def get_address_before_comma(address):
    # Get text before first comma
    parts = address.split(',')
    if parts:
        return parts[0].strip()
    return address

def get_progressive_address_parts(address):
    # Split by comma and clean each part
    parts = [p.strip() for p in address.split(',')]
    addresses = []
    
    # Build progressively larger address strings
    current = parts[0]
    addresses.append(current)
    
    for part in parts[1:]:
        current = f"{current} {part}"  # Remove comma, just use space
        addresses.append(current)
        
    return addresses

# Function to geolocate the address
def geocode_location(address):
    try:
        print(f"\nTrying to geocode address: {address}")
        
        # First try with postal code
        postal_code = extract_postal_code(address)
        if postal_code:
            print(f"1. Attempting with postal code: {postal_code}")
            location = geolocator.geocode(postal_code, country_codes="sg")  # No need for str() since already string
            if location:
                print(f"Success with postal code! Got: {(location.latitude, location.longitude)}")
                return (location.latitude, location.longitude)
        
        # Try progressively with parts of the address
        address_parts = get_progressive_address_parts(address)
        for i, addr in enumerate(address_parts, 2):
            # Skip if address contains only digits
            if addr.strip().isdigit():
                continue
                
            print(f"{i}. Attempting with partial address: {addr}")
            location = geolocator.geocode(addr, country_codes="sg")
            if location:
                print(f"Success with partial address! Got: {(location.latitude, location.longitude)}")
                return (location.latitude, location.longitude)
            
            time.sleep(1)  # Add delay between attempts
        
        print("All attempts failed to get location")
        return None  # Return None if all attempts fail
    except Exception as e:
        print(f"Error geocoding {address}: {str(e)}")
        return None

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
                # Save progress after each row
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
    original_file = '../3. GeocodingPrep/3.1.cleaned_address_out.csv'  # Replace with your original CSV file
    output_file = '4.2.updated_file_with_geolocation.csv'

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