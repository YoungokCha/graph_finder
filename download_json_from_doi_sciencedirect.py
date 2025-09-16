#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chay
"""
import pandas as pd
import requests
import json
import os

# Constants
api_key = API_KEY
inst_token = INST_TOKEN
OUTPUT_DIR = DIR_NAME 

# Function to fetch metadata from Elsevier API
def fetch_metadata(doi, inst_token, api_key):
    request_url = REQUEST_URL
    response = requests.get(request_url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch data for DOI: {doi}. Status Code: {response.status_code}")
        return None
# Function to save JSON to file
def save_json(data, doi):
    doi_cleaned = doi.replace("/", "_")
    filename = f"{doi_cleaned}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f: 
        f.write(data)
       
# Main function to process DOIs from Excel
def process_dois_from_excel(excel_file, sheet_name,inst_token, api_key):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    for doi in df['DOI']:
        print(f"Processing DOI: {doi}")
        metadata = fetch_metadata(doi, inst_token, api_key)
        if metadata:
            save_json(metadata, doi)

if __name__ == "__main__":
    excel_file = FILE_NAME
    sheet_name = SHEET_NAME
    process_dois_from_excel(excel_file, sheet_name,inst_token, api_key)


