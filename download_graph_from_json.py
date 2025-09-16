#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chay
"""
import requests
import os
import re
import xml.etree.ElementTree as ET
from io import BytesIO
import pytesseract
from PIL import Image
import time

pytesseract.pytesseract.tesseract_cmd = TESSERACT_DIR
pattern = r'gr\d+'
clean_text=''
input_folder = INPUT_DIR
output_folder = OUTPUT_DIR
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Function to download image and perform OCR
def download_image(image_url, image_file, caption_data, doi):
    response = requests.get(image_url, stream=True )
    if response.status_code == 200:
            time.sleep(1)
            image = Image.open(BytesIO(response.content))
            ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            ocr_text_list = ocr_result['text']  
            search_terms = [SEARCH_TERM_LIST]
            if (any(term.lower() in (text.lower() for text in ocr_text_list) for term in search_terms) or  any(term in              caption_data.lower() for term in search_terms)):
                counter=image_file.split('-')[-1].split('_')[0]
                with open(output_folder+'/'+doi+'_'+ counter+'.jpg', 'wb') as file:
                    file.write(response.content)
                print(f"Image successfully downloaded after caption & OCR screening {image_file}")
                text_file = image_file.rsplit('.', 1)[0]
                print(text_file)
                caption_text = caption_data.encode('utf-8')
                with open(output_folder+'/'+doi+'_'+ counter+'.txt', 'wb') as file:
                    file.write(caption_text)
            else:
                print("No download after caption & OCR screening")
    else:
        print(f"Failed to download image from {image_url}")

found_files = []
filenames = []
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(input_folder, filename)
        doi =filename.rsplit('.json', 1)[0]
        print(doi)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'xocs': 'http://www.elsevier.com/xml/xocs/dtd'}
            
            for attachment in root.findall('.//xocs:attachment', ns):
                filename = attachment.find('.//xocs:attachment-eid', ns)
                if filename is not None and filename.text.endswith('_lrg.jpg'):
                    image_file = filename.text  
                    print(image_file)
                    match = re.search(pattern, image_file)
                    if match:
                        print(match.group())
                        loc = match.group()
                        ns1 = {'ce':'http://www.elsevier.com/xml/common/dtd'}
                        for figure in root.findall('.//ce:figure', ns1):
                            link = figure.find('ce:link', ns1)
                            if link is not None and link.get('locator') == loc:
                                caption = figure.find('ce:caption/ce:simple-para', ns1)
                                caption_data = ''.join(caption.itertext()) if caption is not None else "No caption"
                                clean_text = ' '.join(caption_data.split())
                    else:
                        print("can't pick up the name for caption")
                    if image_file not in filenames:
                        filenames.append(image_file)
                        image_url = 'https://ars.els-cdn.com/content/image/'+image_file
                        download_image(image_url, image_file, clean_text, doi)
                    else:
                        print(f"Duplicate file '{image_file}' skipped.")
        except ET.ParseError as e:
            print(f"Error parsing {filename}: {e}")
    else:
        continue 
