"""
@author: chay
"""
import os
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_DIR

from pytesseract import Output
import matplotlib.pyplot as plt
from time import sleep

def remove_and_visualize_text(image):
    if image is None:
        print("Error: Image is None.")
        return
    image_with_text_highlighted = image.copy()
    image_with_text_removed = image.copy()
    data = pytesseract.image_to_data(image_with_text_highlighted, output_type=Output.DICT)
    num_boxes = len(data['level'])
    remove_text_margin = 10
    for i in range(num_boxes):
        detected_text = data['text'][i].strip()
        print(detected_text)
        if detected_text and detected_text.lower() not in  [WRONG_DETECTION_LIST]:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(image_with_text_highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image_with_text_removed = cv2.rectangle(image_with_text_removed, (x - remove_text_margin, y - remove_text_margin), (x + w + remove_text_margin, y + h + remove_text_margin), (255, 255, 255), -1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 
    axes[0].imshow(image_with_text_highlighted)
    axes[0].set_title('Text Highlighted')
    axes[0].axis('off')
    axes[1].imshow(image_with_text_removed)
    axes[1].set_title('Text Removed')
    axes[1].axis('off')
    plt.show()
    return image_with_text_removed

def contains_specific_text(image, main_rect, main_rect_cnt):
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(main_rect)
    if main_rect_cnt == 1:
        look_up_margin = 500
    elif main_rect_cnt == 2:
        look_up_margin = 300
    else:
        look_up_margin = 200
    new_rect_x = max(rect_x - look_up_margin, 0)
    new_rect_y = rect_y
    new_rect_w = rect_w + look_up_margin
    new_rect_h = rect_h + look_up_margin
    roi = image[new_rect_y:new_rect_y + new_rect_h, new_rect_x:new_rect_x + new_rect_w]
    text = pytesseract.image_to_string(roi)
    return any(keyword in text for keyword in [WANTED_KEYWORD_LIST]) 
    return True

def is_inside(approx, main_rect):
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(main_rect)
    for point in approx:
        x, y = point[0]
        if not (rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h):
            return False
    return True

def is_duplicate(rect1, rect2):
    x1, y1, w1, h1 = cv2.boundingRect(rect1)
    x2, y2, w2, h2 = cv2.boundingRect(rect2)
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)
    intersection_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    iou = intersection_area / float(rect1_area + rect2_area - intersection_area)
    return iou > 0.9

def detect_and_display_rectangles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image at {image_path}")
    detected_rectangles = []
    main_rectangles = []
    sub_rectangles =[]
    cropped_rectangles = []
    image_area = image.shape[0] * image.shape[1]
    
    max_main_counter = 24
    max_sub_counter = 10
    min_sub_counter = 5
    crop_margin = 25
    text_margin = 30

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = np.median(blurred)
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    edged = cv2.Canny(blurred, lower, upper)
    plt.imshow(edged, cmap='gray')
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.007 * cv2.arcLength(contour, True), True)
        if len(approx) > 2:
            area = cv2.contourArea(contour)
            if area > (image_area / max_main_counter):
                if not any(is_duplicate(approx, existing) for existing in main_rectangles) and not any(is_inside(approx, existing) for existing in main_rectangles):
                    main_rectangles.append(approx)
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
                else:
                    detected_rectangles.append(approx)

    main_counter = len(main_rectangles)
    if main_counter == 1:
        axis_margin = 500
    elif main_counter == 2:
        axis_margin = 250
    else:
        axis_margin = 250

    for main_rect in main_rectangles:
        x, y, w, h = cv2.boundingRect(main_rect)
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        sub_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sub_blurred = cv2.GaussianBlur(sub_gray, (5, 5), 0)
        sub_median = np.median(sub_blurred)
        sub_lower = int(max(0, 0.7 * sub_median))
        sub_upper = int(min(255, 1.3 * sub_median))
        if sub_lower >= sub_upper:
            sub_upper = sub_lower + 1
        sub_edged = cv2.Canny(sub_blurred, sub_lower, sub_upper)
        sub_contours, _ = cv2.findContours(sub_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for sub_contour in sub_contours:
           sub_approx = cv2.approxPolyDP(sub_contour, 0.01 * cv2.arcLength(sub_contour, True), True)
           if len(sub_approx) >=2 :
               area = cv2.contourArea(sub_approx)
               sub_image_area = roi.shape[0] * roi.shape[1]
               if area < (sub_image_area / min_sub_counter) and area > (sub_image_area / max_sub_counter):
                   sub_plot_margin = int(area/8000)
                   x_sub, y_sub, w_sub, h_sub = cv2.boundingRect(sub_approx)                   
                   cv2.rectangle(image, (x + x_sub - sub_plot_margin, y + y_sub), (x + x_sub + w_sub, y + y_sub + h_sub + sub_plot_margin), (255, 255, 255), -1)
                   cv2.rectangle(image, (x + x_sub, y + y_sub), (x + x_sub + w_sub, y + y_sub + h_sub), (255, 0, 255), 10)
                   sub_rectangles.append(sub_approx)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title('Detected Main & Sub plot')
    plt.axis('off')
    plt.show()

    for j, main_rect in enumerate(main_rectangles):
        contains = contains_specific_text(image, main_rect, main_counter)
        if contains:            
            x, y, w, h = cv2.boundingRect(main_rect)
            x_margin = max(x + crop_margin, 0)
            y_margin = max(y + crop_margin, 0)
            w_margin = min(x + w - crop_margin, image.shape[1]) - x_margin
            h_margin = min(y + h - crop_margin, image.shape[0]) - y_margin
            print(x_margin, y_margin, w_margin, h_margin)
            cropped_image = image[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
            cropped_rectangles.append(cropped_image)  

            new_rect_x = max(x - axis_margin, 0)
            new_rect_y = max(y - text_margin , 0)
            new_rect_w = w + axis_margin
            new_rect_h = h + axis_margin - text_margin
            roi_image = image[new_rect_y:new_rect_y + new_rect_h, new_rect_x:new_rect_x + new_rect_w]
            save_path = image_path.split('/')[-1].replace('.jpg', f'_{j + 1}_roi.png')
            cv2.imwrite(save_path, roi_image)
            print(f"Image saved to {save_path}")
            sleep (1)
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            text_remove_image =remove_and_visualize_text(cropped_rgb)
            save_path = image_path.split('/')[-1].replace('.jpg', f'_{j + 1}.png')
            cv2.imwrite(save_path, cv2.cvtColor(text_remove_image, cv2.COLOR_BGR2RGB))
            print(f"Image saved to {save_path}")
    return detected_rectangles, cropped_rectangles



def process_all_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}")
            rectangles, image = detect_and_display_rectangles(image_path)
folder_path = FOLDER_PATH
process_all_images_in_folder(folder_path)