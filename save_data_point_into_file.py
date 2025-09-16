import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import base64
import requests
from sklearn.cluster import DBSCAN
import re
import os
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from scipy.interpolate import make_interp_spline

directory = #folder_path

def sanitize_label(label):
    label = label.strip("'").strip('"')
    #return re.sub(r'[^a-zA-Z0-9\s]', '_', label)
    return label

def create_mask(hsv_image, lower_bound, upper_bound):
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def define_color_ranges():
    color_ranges = {
        "red1": (np.array([0, 50, 50]), np.array([10, 255, 255])),
        "red2": (np.array([170, 50, 50]), np.array([180, 255, 255])),
        "blue": (np.array([100, 120, 50]), np.array([140, 255, 255])),
        "black": (np.array([0, 0, 0]), np.array([100, 255, 50])),
        "purple": (np.array([130, 50, 50]), np.array([160, 255, 255])),
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
        "green": (np.array([40, 50, 50]), np.array([80, 255, 255])),
        "cyan": (np.array([80, 50, 50]), np.array([100, 255, 255])),
        "magenta": (np.array([140, 50, 50]), np.array([170, 255, 255])),
        "orange": (np.array([10, 100, 100]), np.array([20, 255, 255])),
        "pink": (np.array([160, 50, 50]), np.array([170, 255, 255])),
        "light_blue": (np.array([85, 50, 70]), np.array([105, 255, 255])),
    }
    return color_ranges


def detect_and_create_masks(hsv_image, color_name):
    color_ranges = define_color_ranges()
    masks = {}
        
    if color_name in color_ranges:
        lower_bound, upper_bound = color_ranges[color_name]
        mask = create_mask(hsv_image, lower_bound, upper_bound)
        masks[color_name] = mask
        
    elif color_name == "red":
        lower_bound1, upper_bound1 = color_ranges["red1"]
        lower_bound2, upper_bound2 = color_ranges["red2"]
        mask1 = create_mask(hsv_image, lower_bound1, upper_bound1)
        mask2 = create_mask(hsv_image, lower_bound2, upper_bound2)
        masks[color_name] = cv2.bitwise_or(mask1, mask2)

    return masks



def process_mask(mask, color_name, rows, cols, num_rows, num_cols, image_path, legend, axes_data):
    img = np.zeros_like(mask)
    
    cell_width = cols // num_cols
    cell_height = rows // num_rows
    
    x_axis_label, x_min, x_max = axes_data[0]
    y_axis_label, y_min, y_max = axes_data[1]
    
    x_scale = (x_max - x_min) / cols
    y_scale = (y_max - y_min) / rows

    cell_locations = [["Legend", x_axis_label, y_axis_label]]
    cell_locations_file = [["Legend", x_axis_label, y_axis_label]]
    
    x_bins = np.linspace(x_min, x_max, 100)
    detected_points = {x: [] for x in x_bins}

    for i in range(num_rows):
        for j in range(num_cols):
            top = i * cell_height
            bottom = (i + 1) * cell_height
            left = j * cell_width
            right = (j + 1) * cell_width
            
            cell = mask[top:bottom, left:right]
            non_black = np.any(cell > 0)
            
            if non_black:
                for k in range(cell.shape[0]):
                    for l in range(cell.shape[1]):
                        if cell[k, l] > 0:
                            x_value = x_min + (left + l) * x_scale
                            y_value = y_max - (top + k) * y_scale
                            x_bin = min(x_bins, key=lambda b: abs(b - x_value))
                            detected_points[x_bin].append(y_value)
                            break
                    if non_black:
                        break

    
    # Apply outlier filtering for each x_bin with detected points
    filtered_points = []
    for x_bin, y_values in detected_points.items():
        if y_values:
            y_values = np.array(y_values)
            
            if len(y_values) > 2:
                y_z_scores = zscore(y_values)
                y_values_filtered = y_values[np.abs(y_z_scores) < 1.2]  # Filter out points with z-score > 1.5
            else:
                y_values_filtered = y_values  # Skip filtering if insufficient points
            
            if len(y_values_filtered) > 0:
                y_values_filtered.sort()
                middle_y = y_values_filtered[len(y_values_filtered) // 2]
                filtered_points.append((x_bin, middle_y))        ##middle point
            
    # Convert filtered points to arrays
    filtered_points = np.array(filtered_points)
  
    for x_value, y_value in filtered_points:
        cell_locations.append([legend, x_value, y_value])
        col = int((x_value - x_min) / x_scale)
        row = int((y_max - y_value) / y_scale)
        img[row - 2:row + 2, col - 2:col + 2] = 200
        
        if y_max < 10:
            x_value_file = x_value
            y_value_file = 10**(y_value)
        else:    
            x_value_file = x_value 
            y_value_file = y_value

        cell_locations_file.append([legend, x_value_file, y_value_file])
    
    plt.imshow(img, cmap='gist_ncar')
    for i in range(1, num_cols):
        plt.axvline(cols // num_cols * i, color='g', linestyle='--', linewidth=0.5)
    for i in range(1, num_rows):
        plt.axhline(rows // num_rows * i, color='g', linestyle='--', linewidth=0.5)
    plt.axis('off')
    plt.title(f'{color_name.capitalize()} Mask with Grid')
    plt.show()
    
    save_name = image_path.split('/')[-1].replace('.png', '_') + color_name
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    legend_sanitized = re.sub(invalid_chars, '_', legend)
    x_axis_label_sanitized = re.sub(invalid_chars, '_', x_axis_label)
    y_axis_label_sanitized = re.sub(invalid_chars, '_', y_axis_label)
    file_path = f'{save_name}_{legend_sanitized}_{x_axis_label_sanitized}_{y_axis_label_sanitized}.csv'

    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(cell_locations_file)
    print(f"Data has been written to {file_path}")
    
    x_points = [point[1] for point in cell_locations[1:]]
    y_points = [point[2] for point in cell_locations[1:]]
    plt.scatter(x_points, y_points, color=f'{color_name}', s=3)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(f'Detected Points for {legend}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

    
    
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    

def extract_axes_legend_info_using_gpt4o(image_path, api_key):

    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    
    prompt_text = ("Provide the legend colors and names, as well as the axis names and their precise minimum and maximum values from the attached image. If legend is absent, use [(black, *)]. If axis min, max values are in scientific notation (e.g, 10^3), return only the exponent (e.g.,3). Format the output as: Legend: [(color, legend label), (color, legend label), ...] '\n'Axes: [(x-axis_name, x_min, x_max), (y-axis_name, y_min, y_max)]. Provide output strictly in the specified format.")

    payload = {
       "model" : #fine-tuned model
       "messages": [
           { "role": "system", "content": "You are an assistant that extracts the precise graph legend axes data" },           
           {
               "role": "user",
               "content": [
                   {"type" : "text",
                    "text" : prompt_text },
                   {"type": "image_url",
                    "image_url":{ "url": f"data:image/jpeg;base64,{base64_image}",
                                 "detail": "high" }
                    }
                   ]
           }
       ],
   }

    try:
       response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
       response.raise_for_status()
       result = response.json()
       
       # Extracting content from the response
       response_content = result['choices'][0]['message']['content']
       
       return response_content

    except requests.exceptions.RequestException as e:
       print(f"Request failed: {e}")
       return ""
    except KeyError:
       print("Unexpected response structure.")
       return ""


def main(image_path, legends, axes_data):
    image_origin = cv2.imread(image_path)
    image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.title('BGR image')
    plt.show() 
    
    hsv_image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_image)
    plt.axis('off')
    plt.title('HSV image')
    plt.show() 
    
    rows, cols = image.shape[0], image.shape[1]
    print(rows, cols)
    num_rows, num_cols = int(rows/2), int(cols/2)
    
    for color_name, legend in legends:
        masks = detect_and_create_masks(hsv_image, color_name)
        for mask_color_name, mask in masks.items():
            process_mask(mask, mask_color_name, rows, cols, num_rows, num_cols, image_path, legend, axes_data)
          
def parse_color_label_pairs(s):
    print(s)
    #return re.findall(r"\(\s*(['\"]?[^,'\"]+['\"]?)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", s)
    return re.findall(r'\((\w+),\s*([^)]+)\)', s)

def parse_label_min_max(s):
    print(s)
    matches = re.findall(r"\(\s*(['\"]?[^,'\"]+['\"]?)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", s)
    
    def extract_exponent(value):
        match = re.match(r"10[⁰¹²³⁴⁵⁶⁷⁸⁹]+", value)
        if match:
            exponent_str = value[2:]
            exponent = ''.join(str("⁰¹²³⁴⁵⁶⁷⁸⁹".index(c)) for c in exponent_str if c in "⁰¹²³⁴⁵⁶⁷⁸⁹")
            return int(exponent)
        try:
            return eval(value)
        except (SyntaxError, NameError):
            return value  # Return as string if evaluation fails
    
    results = [[m[0], extract_exponent(m[1]), extract_exponent(m[2])] for m in matches]
    
    return results


if __name__ == "__main__":
    api_key = #API KEY 
    
    roi_files = {}
    regular_files = {}

    for filename in os.listdir(directory):
        if filename.endswith('_roi.png'):
            base_name = filename[:-8]
            roi_files[base_name] = os.path.join(directory, filename)
        elif filename.endswith('.png'):
            base_name = filename[:-4]
            regular_files[base_name] = os.path.join(directory, filename)

    for base_name in roi_files:
        if base_name in regular_files:
            image_path_roi = roi_files[base_name]
            image_path = regular_files[base_name]
            
            results = extract_axes_legend_info_using_gpt4o(image_path_roi, api_key)
            parts = results.split('\n')
            
            if len(parts) < 2:
                print(f"Error: The results for {image_path_roi} are not in the expected format.")
                continue
            
            # Parse legend color-label pairs and axis data with min/max values
            color_label_pairs = parse_color_label_pairs(parts[0])
            label_min_max_pairs = parse_label_min_max(parts[1])
      
            if color_label_pairs and label_min_max_pairs:  
                colors = [pair[0] for pair in color_label_pairs]
                labels_from_colors = [sanitize_label(pair[1]) for pair in color_label_pairs]
                labels_from_values = [sanitize_label(pair[0]) for pair in label_min_max_pairs]
               
                #print(labels_from_values)
                min_numbers = [pair[1] for pair in label_min_max_pairs]                
                max_numbers = [pair[2] for pair in label_min_max_pairs]

                if all(is_number(min_val) for min_val in min_numbers) and all(is_number(max_val) for max_val in max_numbers):
                    min_numbers = [float(min_val) for min_val in min_numbers]
                    max_numbers = [float(max_val) for max_val in max_numbers]

                    legends = list(zip(colors, labels_from_colors))
                    axes_data = list(zip(labels_from_values, min_numbers, max_numbers))
                
                    print("Legends:", legends)
                    print("Axes Data:", axes_data)
                
                    main(image_path, legends, axes_data)
                else:
                    print(f"Error: Non-numeric min/max values found in {image_path_roi}, skipping this file.")
            else:
                print(f"Error: Color label pairs or label min-max pairs are empty for {image_path_roi}.")
