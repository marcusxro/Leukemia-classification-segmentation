import os
import sys
import argparse
import glob
import time
import csv
from datetime import datetime
import math

import cv2
import numpy as np
from ultralytics import YOLO
from skimage import feature, measure, morphology, filters
import pandas as pd
from collections import deque

# ==============================================
# Medical Configuration Parameters
# ==============================================

CLASSES = {
    0: 'Benign',
    1: 'Early Stage',
    2: 'Pre',
    3: 'Pro'
}

MEDICAL_NOTES = {
    'Benign': 'Normal blood cells detected. No signs of malignancy.',
    'Early Stage': 'Early signs of abnormal cell morphology (N:C ratio > 0.8, mild irregularity). Recommend follow-up testing.',
    'Pre': 'Pre-leukemic conditions detected (N:C ratio > 1.0, visible nuclear abnormalities). CBC and flow cytometry recommended.',
    'Pro': 'Leukemic blast cells detected (N:C ratio > 1.2, marked nuclear irregularity). Immediate hematology consultation advised.'
}


COLORS = {
    'Benign': (76, 175, 80),      
    'Early Stage': (255, 193, 7), 
    'Pre': (255, 152, 0),        
    'Pro': (244, 67, 54)         
}


# ==============================================
# Core Functions
# ==============================================


MAX_SEGMENTATION_PANELS = 16  
SEGMENTATION_GRID_SIZE = (800, 800) 


def segment_nucleus(cell_roi):
    """Improved nucleus segmentation focusing on dark violet center"""

    gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
    

    thresh = cv2.adaptiveThreshold(gray, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 51, 10)
    
    
    hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
    lower_violet = np.array([120, 40, 40])
    upper_violet = np.array([160, 255, 255])
    violet_mask = cv2.inRange(hsv, lower_violet, upper_violet)
    
 
    combined = cv2.bitwise_or(thresh, violet_mask)
    
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        nucleus_mask = np.zeros_like(gray)
        cv2.drawContours(nucleus_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
   
        if cv2.contourArea(largest_contour) > 50:
            return nucleus_mask
    
    return np.zeros_like(gray)

def segment_cytoplasm(cell_roi, nucleus_mask):
    """Segment cytoplasm with medical visualization in mind"""

    gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
    

    thresh = cv2.adaptiveThreshold(gray, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 51, 10)
    
  
    thresh[nucleus_mask > 0] = 0
    
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
   
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cytoplasm_mask = (labels == largest_label).astype(np.uint8) * 255
        
   
        if np.sum(cytoplasm_mask) > 100:
            return cytoplasm_mask
    
    return np.zeros_like(thresh)


def calculate_morphological_features(cell_roi, nucleus_mask, cytoplasm_mask):
    """Calculate features with robust error handling"""
    features = {
        'cell_area': 0,
        'nucleus_area': 0,
        'cytoplasm_area': 0,
        'nc_ratio': 0,
        'perimeter': 0,
        'circularity': 0,
        'irregularity': 0,
        'convexity': 0,
        'cytoplasm_perimeter': 0
    }

    try:
       
        features['nucleus_area'] = np.sum(nucleus_mask > 0)
        features['cytoplasm_area'] = np.sum(cytoplasm_mask > 0)
        features['cell_area'] = features['nucleus_area'] + features['cytoplasm_area']
        features['nc_ratio'] = features['nucleus_area'] / (features['cell_area'] + 1e-6)
        
       
        contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            features['perimeter'] = cv2.arcLength(cnt, True)
            features['circularity'] = (4 * math.pi * features['nucleus_area']) / (features['perimeter']**2 + 1e-6)
            features['irregularity'] = features['perimeter'] / (2 * math.sqrt(math.pi * features['nucleus_area']))
            hull = cv2.convexHull(cnt)
            features['convexity'] = cv2.contourArea(hull) / (features['nucleus_area'] + 1e-6)
        
       
        cyto_contours, _ = cv2.findContours(cytoplasm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cyto_contours:
            cnt = max(cyto_contours, key=cv2.contourArea)
            features['cytoplasm_perimeter'] = cv2.arcLength(cnt, True)
            
    except Exception as e:
        print(f"Error calculating features: {e}")

    return features


def create_segmentation_grid(segmented_cells, panel_width, panel_height):
    """Create properly sized grid for medical viewing"""
    if not segmented_cells:
         return np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    num_cells = len(segmented_cells)
    cols = min(4, num_cells)  
    rows = math.ceil(num_cells / cols)
    
 
    cell_size = min(200, panel_width // (cols + 1), panel_height // (rows + 1))
    
   
    grid = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
   
    for i, cell_img in enumerate(segmented_cells):
        row = i // cols
        col = i % cols
        
       
        resized = cv2.resize(cell_img, (cell_size, cell_size))
        

        y_start = row * (cell_size + 10) + 10
        x_start = col * (cell_size + 10) + 10
        
       
        if y_start + cell_size <= panel_height and x_start + cell_size <= panel_width:
            grid[y_start:y_start+cell_size, x_start:x_start+cell_size] = resized
    
    return grid


def calculate_texture_features(cell_roi):
    """Calculate texture features with error handling"""
    features = {
        'contrast': 0,
        'energy': 0,
        'homogeneity': 0,
        'lbp_entropy': 0
    }
    
    try:
        gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        glcm = feature.graycomatrix(gray, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2], symmetric=True, normed=True)
        features['contrast'] = np.mean(feature.graycoprops(glcm, 'contrast'))
        features['energy'] = np.mean(feature.graycoprops(glcm, 'energy'))
        features['homogeneity'] = np.mean(feature.graycoprops(glcm, 'homogeneity'))
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        features['lbp_entropy'] = measure.shannon_entropy(lbp)
    except Exception as e:
        print(f"Error calculating texture: {e}")
    
    return features

def calculate_color_features(cell_roi):
    """Calculate color features with error handling"""
    features = {
        'hue_mean': 0, 'hue_std': 0,
        'saturation_mean': 0, 'saturation_std': 0,
        'value_mean': 0, 'value_std': 0,
        'color_heterogeneity': 0
    }
    
    try:
        hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            features[f'{channel}_mean'] = np.mean(hsv[:,:,i])
            features[f'{channel}_std'] = np.std(hsv[:,:,i])
        features['color_heterogeneity'] = np.std([features['hue_std'], features['saturation_std'], features['value_std']])
    except Exception as e:
        print(f"Error calculating color: {e}")
    
    return features

def visualize_segmentation_medical(cell_roi, nucleus_mask):
    """Dark cell with white nuclear details visualization"""
    
    gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    vis = np.zeros_like(cell_roi)
    
   
    nuclear_details = cv2.bitwise_and(gray, gray, mask=nucleus_mask)
    _, details_thresh = cv2.threshold(nuclear_details, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
   
    vis[details_thresh > 0] = [255, 255, 255]
    
    
    contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(vis, contours, -1, (200, 200, 200), 1)
    
    return vis


def generate_medical_report(class_counts, total_cells, features_df, patient_id):
    """Generate report with error handling for missing features"""
    report_lines = [
        "\n=== HEMATOLOGICAL ANALYSIS REPORT ===",
        f"Patient ID: {patient_id}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nTotal Cells Analyzed: {total_cells}",
        "\nCell Classification Summary:"
    ]
    
    percentages = {}
    for class_id, count in class_counts.items():
        percentage = (count / total_cells) * 100 if total_cells > 0 else 0
        class_name = CLASSES.get(class_id, f"Unknown_{class_id}")
        percentages[class_name] = percentage
        report_lines.append(f"{class_name}: {count} cells ({percentage:.1f}%)")
    
    try:
        if not features_df.empty:
            avg_nc = features_df.get('nc_ratio', pd.Series([0])).mean()
            avg_irregularity = features_df.get('irregularity', pd.Series([0])).mean()
            blast_percentage = percentages.get('Pro', 0)
            
            report_lines.extend([
                f"\nKey Morphological Indicators:",
                f"Average Nuclear-Cytoplasmic Ratio: {avg_nc:.2f}",
                f"Average Nuclear Irregularity Index: {avg_irregularity:.2f}",
                f"Blast Cell Percentage: {blast_percentage:.1f}%"
            ])
    except Exception as e:
        report_lines.append(f"\nWarning: Could not calculate all features ({str(e)})")
    
    if class_counts.get(3, 0) > 0:
        diagnosis = "Pro"
    elif class_counts.get(2, 0) > 0:
        diagnosis = "Pre"
    elif class_counts.get(1, 0) > 0:
        diagnosis = "Early Stage"
    else:
        diagnosis = "Benign"
    
    report_lines.extend([
        f"\nDIAGNOSIS: {diagnosis}",
        f"MEDICAL NOTES: {MEDICAL_NOTES.get(diagnosis, 'No notes available')}",
        "\n=== END REPORT ==="
    ])
    
    return diagnosis, "\n".join(report_lines)

# ==============================================
# Main Processing Pipeline
# ==============================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Leukemia Classification System')
    parser.add_argument('--model', required=True, help='Path to YOLO model file')
    parser.add_argument('--source', required=True, help='Image source (file, folder, or camera)')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--resolution', help='Display resolution WxH')
    parser.add_argument('--save', action='store_true', help='Save output images/video')
    parser.add_argument('--save_csv', action='store_true', help='Save cell data to CSV')
    parser.add_argument('--patient_id', default='UNKNOWN', help='Patient identifier')
    parser.add_argument('--verbose', action='store_true', help='Show detailed processing info')
    return parser.parse_args()

def initialize_video_source(source, resolution=None):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_type = 'camera'
    elif os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        source_type = 'video'
    else:
        raise ValueError("Invalid video source")
    
    if resolution:
        w, h = map(int, resolution.split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    return cap, source_type

def process_image(frame, model, args):
    """Processing with fixed dimensions to avoid broadcast error"""
    try:
        
        original_frame = frame.copy()
        results = model(frame, verbose=args.verbose)
        detections = results[0].boxes

        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        features_data = []
        segmented_cells = []
        cell_id = 0
        
        for i in range(len(detections)):
            try:
                xyxy = detections[i].xyxy.cpu().numpy().squeeze()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                class_id = int(detections[i].cls.item())
                confidence = detections[i].conf.item()
                
                if confidence > args.thresh:
                    cell_id += 1
                    class_counts[class_id] += 1
                    
                    
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(frame.shape[1], xmax)
                    ymax = min(frame.shape[0], ymax)
                    
                    
                    if xmax <= xmin or ymax <= ymin:
                        continue
                        
                    cell_roi = frame[ymin:ymax, xmin:xmax]
                    if cell_roi.size == 0:
                        continue
                        
                    nucleus_mask = segment_nucleus(cell_roi)
                    
                    
                    seg_vis = visualize_segmentation_medical(cell_roi, nucleus_mask)
                    
                   
                    cv2.putText(seg_vis, f"ID:{cell_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    segmented_cells.append(seg_vis)
                    
                    
                    cytoplasm_mask = segment_cytoplasm(cell_roi, nucleus_mask)
                    morph_features = calculate_morphological_features(cell_roi, nucleus_mask, cytoplasm_mask)
                    
                    cell_features = {
                        'cell_id': cell_id,
                        'class': CLASSES.get(class_id, f"Unknown_{class_id}"),
                        'confidence': confidence,
                        **morph_features
                    }
                    features_data.append(cell_features)
                    
                   
                    class_name = CLASSES.get(class_id, f"Unknown_{class_id}")
                    color = COLORS.get(class_name, (255, 255, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f"ID:{cell_id} {class_name} {confidence*100:.1f}%"
                    cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue
        
       
        h, w = frame.shape[:2]
        
        
        square_size = min(h, w)
        right_panel_width = square_size
        output_height = square_size
        
        
        left_panel_width = w - right_panel_width
        left_panel = np.zeros((output_height, left_panel_width, 3), dtype=np.uint8)
        
        
        if segmented_cells:
            cols = min(4, len(segmented_cells))
            rows = math.ceil(len(segmented_cells) / cols)
            cell_size = min(200, left_panel_width // (cols + 1), output_height // (rows + 1))
            
            for i, cell_img in enumerate(segmented_cells):
                row = i // cols
                col = i % cols
                y = 10 + row * (cell_size + 10)
                x = 10 + col * (cell_size + 10)
                
                if y + cell_size < output_height and x + cell_size < left_panel_width:
                    
                    resized = cv2.resize(cell_img, (cell_size, cell_size))
                    left_panel[y:y+cell_size, x:x+cell_size] = resized
        
        
        right_panel = np.zeros((output_height, right_panel_width, 3), dtype=np.uint8)
        
        
        center_x, center_y = w // 2, h // 2
        half_size = square_size // 2
        cropped_frame = original_frame[
            max(0, center_y - half_size):min(h, center_y + half_size),
            max(0, center_x - half_size):min(w, center_x + half_size)
        ]
        
        
        resized_frame = cv2.resize(cropped_frame, (right_panel_width, output_height))
        right_panel[:, :] = resized_frame
        
        
        for i in range(len(detections)):
            try:
                xyxy = detections[i].xyxy.cpu().numpy().squeeze()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                class_id = int(detections[i].cls.item())
                confidence = detections[i].conf.item()
                
                if confidence > args.thresh:
                   
                    rxmin = xmin - (center_x - half_size)
                    rymin = ymin - (center_y - half_size)
                    rxmax = xmax - (center_x - half_size)
                    rymax = ymax - (center_y - half_size)
                    
                    
                    if (0 <= rxmin < right_panel_width and 0 <= rymin < output_height and
                        0 <= rxmax < right_panel_width and 0 <= rymax < output_height):
                        
                        class_name = CLASSES.get(class_id, f"Unknown_{class_id}")
                        color = COLORS.get(class_name, (255, 255, 255))
                        cv2.rectangle(right_panel, (rxmin, rymin), (rxmax, rymax), color, 2)
                        label = f"ID:{i+1} {class_name} {confidence*100:.1f}%"
                        cv2.putText(right_panel, label, (rxmin, rymin-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            except Exception as e:
                print(f"Error drawing detection {i} on right panel: {e}")
                continue
        
        
        total_cells = sum(class_counts.values())
        if total_cells > 0:
            features_df = pd.DataFrame(features_data)
            diagnosis, report = generate_medical_report(class_counts, total_cells, features_df, args.patient_id)
            
            
            text_x, text_y = 20, 30
            line_height = 25
            cv2.putText(right_panel, f"Diagnosis: {diagnosis}", (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(right_panel, f"Blast Cells: {class_counts.get(3, 0)}", 
                       (text_x, text_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(right_panel, f"Total Cells: {total_cells}", 
                       (text_x, text_y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if args.verbose:
                print(report)
        
        
        combined = np.zeros((output_height, left_panel_width + right_panel_width, 3), dtype=np.uint8)
        combined[:, :left_panel_width] = left_panel
        combined[:, left_panel_width:] = right_panel
        
        return combined, features_data

    except Exception as e:
        print(f"Error processing image: {e}")
       
        return frame, []



def main():
    args = parse_arguments()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)

    try:
        model = YOLO(args.model, task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

   
    if args.save and not os.path.exists('output'):
        os.makedirs('output')

    
    csv_file, csv_writer = None, None
    if args.save_csv:
        try:
            csv_file = open(f"leukemia_analysis_{args.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 'w')
            fieldnames = [
                'patient_id', 'timestamp', 'cell_id', 'class', 'confidence',
                'cell_area', 'nucleus_area', 'nc_ratio', 'perimeter', 'circularity', 
                'irregularity', 'convexity', 'cytoplasm_area', 'cytoplasm_perimeter'
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            args.save_csv = False

    try:
        if os.path.isdir(args.source):
            image_files = glob.glob(os.path.join(args.source, '*.*'))
            for img_path in image_files:
                try:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        print(f"Could not read image {img_path}")
                        continue
                        
                    processed_frame, features = process_image(frame, model, args)
                    
                    if args.save_csv and features:
                        for feat in features:
                            feat.update({
                                'patient_id': args.patient_id,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            csv_writer.writerow(feat)
                    
                    if args.save:
                        output_path = os.path.join('output', os.path.basename(img_path))
                        cv2.imwrite(output_path, processed_frame)
                    
                    cv2.imshow('Leukemia Analysis - Left: Segmented Cells | Right: Detection', processed_frame)
                    if cv2.waitKey(0) == ord('q'):
                        break
                
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue
        else:
            cap, source_type = initialize_video_source(args.source, args.resolution)
            
            if args.save and source_type == 'video':
                try:
                  
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (width, height))
                except Exception as e:
                    print(f"Error creating video writer: {e}")
                    args.save = False
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame, features = process_image(frame, model, args)
                    
                    if args.save_csv and features:
                        for feat in features:
                            feat.update({
                                'patient_id': args.patient_id,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            csv_writer.writerow(feat)
                    
                    if args.save and source_type == 'video':
                        
                        output_frame = cv2.resize(processed_frame, (width, height))
                        out.write(output_frame)
                    
                    cv2.imshow('Leukemia Analysis - Left: Segmented Cells | Right: Detection', processed_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                except Exception as e:
                    print(f"Error processing video frame: {e}")
                    continue
            
            cap.release()
            if args.save and source_type == 'video':
                out.release()

    except Exception as e:
        print(f"Fatal error: {e}")

    finally:
        if args.save_csv and csv_file:
            csv_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()