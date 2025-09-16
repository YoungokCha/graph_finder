"""
@author: chay
"""
import os
import re
import csv
import base64
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.stats import zscore

directory = DIRECTORY

def sanitize_label(label):
    return label.strip("'").strip('"')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def convert_to_float(s):
    s = s.strip()
    if "*10^" in s or "×10^" in s:
        s_mod = s.replace("*10^", "e").replace("×10^", "e")
        try:
            return float(s_mod)
        except ValueError:
            print(f"DEBUG: Could not convert '{s}' (modified: '{s_mod}')")
            return None
    elif "^" in s:
        match = re.fullmatch(r"([-+]?\d*\.?\d+)\^([-+]?\d+)", s)
        if match:
            base_str, exp_str = match.groups()
            try:
                return float(base_str) ** float(exp_str)
            except ValueError:
                print(f"DEBUG: Could not convert base/exponent in '{s}'")
                return None
        else:
            try:
                return float(s)
            except ValueError:
                print(f"DEBUG: Could not convert '{s}'")
                return None
    else:
        try:
            return float(s)
        except ValueError:
            print(f"DEBUG: Could not convert '{s}'")
            return None

def create_mask(hsv_image, lower_bound, upper_bound):
    return cv2.inRange(hsv_image, lower_bound, upper_bound)

def define_color_ranges():
    return {
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

def detect_and_create_masks(hsv_image, color_name):
    color_ranges = define_color_ranges()
    masks = {}
    if color_name in color_ranges:
        lower, upper = color_ranges[color_name]
        masks[color_name] = create_mask(hsv_image, lower, upper)
    elif color_name == "red":
        lower1, upper1 = color_ranges["red1"]
        lower2, upper2 = color_ranges["red2"]
        mask1 = create_mask(hsv_image, lower1, upper1)
        mask2 = create_mask(hsv_image, lower2, upper2)
        masks[color_name] = cv2.bitwise_or(mask1, mask2)
    return masks

def parse_legend(line):
    try:
        _, content = line.split(":", 1)
    except ValueError:
        return None
    content = content.strip().strip("[]")
    pairs = re.findall(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", content)
    return pairs

def parse_axes(line):
    try:
        _, content = line.split(":", 1)
    except ValueError:
        return None
    content = content.strip().strip("[]")
    parts = [p.strip() for p in content.split(",")]
    return parts if len(parts) == 2 else None

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def extract_axes_legend_info_using_gpt4o(image_path, api_key):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_text = PROMPT_TEXT
    payload = {
        "model": GPT_MODEL,
        "messages": [
            { "role": "system", "content": "You are an assistant that extracts the precise graph legend axes data" },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Request failed or unexpected response: {e}")
        return ""
    
def process_mask(mask, color_name, rows, cols, num_rows, num_cols, image_path, legend, axes_data,caption_text):
    img = np.zeros_like(mask).
    x_axis_label, x_min, x_max = axes_data[0]
    y_axis_label, y_min, y_max = axes_data[1]
    x_scale = (x_max - x_min) / cols
    if y_min == 0:
        is_log_scale = False
    else:
        is_log_scale = (y_max / y_min) > 1000 
    if is_log_scale:
        log_y_max = np.log10(y_max)
        log_y_min = np.log10(y_min)
        y_scale_log = (log_y_max - log_y_min) / rows
    else:
        y_scale = (y_max - y_min) / rows
    cell_locations = [["Legend", x_axis_label, y_axis_label]]
    cell_locations_file = [["Legend", x_axis_label, y_axis_label]]
    x_bins = np.linspace(x_min, x_max, 100)
    detected_points = {bin_val: [] for bin_val in x_bins}
    cell_width = cols // num_cols
    cell_height = rows // num_rows

    for i in range(num_rows):
        for j in range(num_cols):
            top = i * cell_height
            bottom = (i + 1) * cell_height
            left = j * cell_width
            right = (j + 1) * cell_width
            cell = mask[top:bottom, left:right]
            indices = np.argwhere(cell > 0)
            if indices.size > 0:
                k, l = indices[0]
                x_value = x_min + (left + l) * x_scale
                pixel_row = top + k
                if is_log_scale:
                    log_y_value = np.log10(y_max) - (pixel_row/rows) * (np.log10(y_max) - np.log10(y_min))
                    y_value = 10 ** log_y_value
                else:
                    y_value = y_max - (pixel_row / rows) * (y_max - y_min)
                x_bin = min(x_bins, key=lambda b: abs(b - x_value))
                detected_points[x_bin].append(y_value)
    filtered_points = []
    for x_bin, y_values in detected_points.items():
        if y_values:
            y_arr = np.array(y_values)
            if len(y_arr) > 2:
                median = np.median(y_arr)
                mad = np.median(np.abs(y_arr - median))
                if mad < 1e-6:
                    y_filtered = y_arr
                else:
                    y_filtered = y_arr[np.abs(y_arr - median) / mad < 0.2]
            else:
                y_filtered = y_arr
            if len(y_filtered) > 0:
                median_y = np.median(np.sort(y_filtered))
                filtered_points.append((x_bin, median_y))
    filtered_points = np.array(filtered_points)
    
    for x_val, y_val in filtered_points:
        cell_locations.append([legend, x_val, y_val])
        col = int((x_val - x_min) / x_scale)
        if is_log_scale:
            row = int((log_y_max - np.log10(y_val)) / (log_y_max - log_y_min) * rows)
        else:
            row = int((y_max - y_val) / (y_max - y_min) * rows)
        img[max(0, row-2):min(rows, row+2), max(0, col-2):min(cols, col+2)] = 200
        cell_locations_file.append([legend, x_val, y_val])
    plt.figure()
    plt.imshow(img, cmap='gist_ncar')

    x_grid_data = np.linspace(x_min, x_max, num_cols + 1)
    x_grid_pixels = (x_grid_data - x_min) / (x_max - x_min) * cols
    for x_pix in x_grid_pixels:
        plt.axvline(x_pix, color='g', linestyle='--', linewidth=0.5)

    if is_log_scale:
        y_grid_data = np.logspace(np.log10(y_min), np.log10(y_max), num_rows + 1)
        y_grid_pixels = (log_y_max - np.log10(y_grid_data)) / (log_y_max - log_y_min) * rows
    else:
        y_grid_data = np.linspace(y_min, y_max, num_rows + 1)
        y_grid_pixels = (y_max - y_grid_data) / (y_max - y_min) * rows
    for y_pix in y_grid_pixels:
        plt.axhline(y_pix, color='g', linestyle='--', linewidth=0.5)
    plt.axis('off')
    plt.title(f'{color_name.capitalize()} Mask with Data Grid')
    plt.show()
    
    save_name = os.path.basename(image_path).replace('.png', f'_{color_name}')
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    legend_sanitized = re.sub(invalid_chars, '_', legend)
    x_axis_label_sanitized = re.sub(invalid_chars, '_', x_axis_label)
    y_axis_label_sanitized = re.sub(invalid_chars, '_', y_axis_label)
    caption_text_sanitized = re.sub(invalid_chars, '_', caption_text)

    file_name = f'{legend_sanitized}_{x_axis_label_sanitized}_{y_axis_label_sanitized}_{caption_text_sanitized}.csv'
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(cell_locations_file)
    print(f"Data has been written to {file_path}")

    plt.figure()
    x_points = [pt[1] for pt in cell_locations[1:]]
    y_points = [pt[2] for pt in cell_locations[1:]]
    plt.scatter(x_points, y_points, color=color_name, s=3)
   
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if is_log_scale:
        plt.yscale('log')
    plt.title(f'Detected Points for {legend}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def main(image_path, legends, axes_data, caption_text):
    image_origin = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('RGB Image')
    plt.show()

    hsv_image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_image)
    plt.axis('off')
    plt.title('HSV Image')
    plt.show()

    rows, cols = image_origin.shape[:2]
    num_rows, num_cols = rows // 2, cols // 2

    for color_name, legend in legends:
        masks = detect_and_create_masks(hsv_image, color_name)
        for mask in masks.values():
            process_mask(mask, color_name, rows, cols, num_rows, num_cols, image_path, legend, axes_data, caption_text)

if __name__ == "__main__":

    api_key = API_KEY

    roi_files, regular_files, text_files = {}, {}, {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('_roi.png'):
            base_name = filename[:-8]
            roi_files[base_name] = filepath
        elif filename.endswith('.png'):
            base_name = filename[:-4]
            regular_files[base_name] = filepath
        elif filename.endswith('_sum.txt'):
            base_name = filename[:-8]
            text_files[base_name] = filepath

    for base_name in roi_files:
        if base_name in regular_files:
            image_path_roi = roi_files[base_name]
            image_path = regular_files[base_name]
            results = extract_axes_legend_info_using_gpt4o(image_path_roi, api_key)
            parts = results.split('\n')
            if len(parts) < 2:
                print(f"Error: The results for {image_path_roi} are not in the expected format.")
                continue

            legend_line, axes_line = parts[0], parts[1]
            legend_pairs = parse_legend(legend_line)
            if not legend_pairs:
                print("Error: Could not parse the legend line.")
                continue

            colors = [pair[0].strip() for pair in legend_pairs]
            labels_from_colors = [sanitize_label(pair[1]) for pair in legend_pairs]
            axes_labels = parse_axes(axes_line)
            if not axes_labels or len(axes_labels) != 2:
                print(f"Error: Could not parse the axes line for {image_path_roi}.")
                continue

            if base_name in text_files:
                text_file_path = text_files[base_name]
                with open(text_file_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if len(lines) < 4:
                    print(f"Error: The text file {text_file_path} does not contain enough data.")
                    continue

                min_y = convert_to_float(lines[0])
                max_y = convert_to_float(lines[1])
                min_x = convert_to_float(lines[2])
                max_x = convert_to_float(lines[3])
                caption_text = lines[4]
                if None in (min_x, max_x, min_y, max_y):
                    print(f"Error: Could not convert one or more min/max values in {text_file_path}")
                    continue

                min_numbers = [min_x, min_y]
                max_numbers = [max_x, max_y]
            else:
                print(f"Warning: No corresponding text file found for {base_name}. Skipping.")
                continue

            legends = list(zip(colors, labels_from_colors))
            axes_data = list(zip(axes_labels, min_numbers, max_numbers))
            print("Legends:", legends)
            print("Axes Data:", axes_data)
            main(image_path, legends, axes_data, caption_text)
