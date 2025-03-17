#!/usr/bin/env python
# evaluate.py - Evaluation script for Music Sheet OMR model

import os
import glob
import argparse
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
import yaml
from tqdm import tqdm
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OMR model on test images')
    parser.add_argument('--model', type=str, default='output/models/best.pt', help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.15, help='Confidence threshold for detections')
    parser.add_argument('--test-dir', type=str, default='raw_data/images/val', help='Directory of test images')
    parser.add_argument('--output-dir', type=str, default='output/evaluation', help='Directory to save results')
    parser.add_argument('--class-map', type=str, default='raw_data/class_mapping.csv', help='Path to class mapping CSV')
    parser.add_argument('--save-json', action='store_true', help='Save detection results as JSON')
    parser.add_argument('--img-size', type=int, default=1024, help='Image size for inference')
    return parser.parse_args()


def load_model(weights_path):
    """Load YOLOv5 model for inference"""
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return None

    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        print(f"Model loaded successfully from {weights_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_class_mapping(class_map_path):
    """Load class mapping from CSV file"""
    if not os.path.exists(class_map_path):
        print(f"Error: Class mapping file not found at {class_map_path}")
        return None

    try:
        class_map = pd.read_csv(class_map_path)

        # Create dictionaries for mapping
        id_to_name = dict(zip(class_map['original_id'], class_map['class_name']))
        yolo_to_id = dict(zip(class_map['yolo_id'], class_map['original_id']))
        yolo_to_name = dict(zip(class_map['yolo_id'], class_map['class_name']))

        return {
            'id_to_name': id_to_name,
            'yolo_to_id': yolo_to_id,
            'yolo_to_name': yolo_to_name
        }
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        return None


def run_inference(model, image_path, conf_threshold, class_mapping):
    """Run inference on a single image and return detection data"""
    if model is None:
        print("Error: Model not loaded")
        return None, None

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    # Set confidence threshold
    model.conf = conf_threshold

    # Run inference
    start_time = time.time()
    results = model(image_path)
    inference_time = time.time() - start_time

    # Get image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Extract detection data
    detection_data = []

    # Process each detection
    for pred in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = pred

        # Get class name if mapping is available
        class_name = f"class_{int(cls_id)}"
        original_class_id = int(cls_id)

        if class_mapping and 'yolo_to_id' in class_mapping and 'yolo_to_name' in class_mapping:
            if int(cls_id) in class_mapping['yolo_to_id']:
                original_class_id = class_mapping['yolo_to_id'][int(cls_id)]

            if int(cls_id) in class_mapping['yolo_to_name']:
                class_name = class_mapping['yolo_to_name'][int(cls_id)]

        # Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Add to detection data
        detection_data.append({
            'class_id': original_class_id,
            'class_name': class_name,
            'confidence': float(conf),
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'x_center': float(center_x),
            'y_center': float(center_y),
            'width': float(x2 - x1),
            'height': float(y2 - y1),
            'center_x': float(center_x / width),
            'center_y': float(center_y / height)
        })

    # Sort by x-position (left to right)
    detection_data.sort(key=lambda x: x['x_center'])

    return detection_data, results, inference_time


def visualize_detection_with_labels(image_path, detection_data, output_path=None, figsize=(15, 10)):
    """Visualize detected symbols with their class labels and save to file"""
    if not detection_data:
        print("No detection data available")
        return None

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Create copy for drawing
    vis_img = img.copy()

    # Sort detections by x position (left to right)
    detection_data.sort(key=lambda x: x['x_center'])

    # Draw each detection with its class label
    for det in detection_data:
        # Get coordinates
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])

        # Draw bounding box (GREEN)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label text
        label = det['class_name']
        conf = f"{det['confidence']:.2f}"
        display_label = f"{label} ({conf})"

        if len(display_label) > 20:
            display_label = display_label[:17] + "..."

        # Calculate text position
        text_x = max(5, min(x1, w-100))
        text_y = max(25, min(y1-10, h-10))

        # Draw label with background
        text_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_img, (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
        cv2.putText(vis_img, display_label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # If output path provided, save the image
    if output_path:
        plt.figure(figsize=figsize)
        plt.imshow(vis_img)
        plt.title(f"Detected Music Notation with Labels ({len(detection_data)} symbols)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Also save as PNG using OpenCV (without matplotlib modifications)
        cv2.imwrite(output_path.replace('.png', '_cv2.png'),
                   cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    return vis_img


def analyze_detections(detection_data):
    """Analyze detection results and provide statistics"""
    if not detection_data:
        print("No detection data available")
        return None

    # Create DataFrame
    df = pd.DataFrame(detection_data)

    # Count detections per class
    class_counts = df['class_name'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']

    # Get confidence statistics
    confidence_stats = df['confidence'].describe()

    # Analyze spatial distribution
    spatial_stats = {
        'x_min': df['center_x'].min(),
        'x_max': df['center_x'].max(),
        'y_min': df['center_y'].min(),
        'y_max': df['center_y'].max(),
    }

    return {
        'total_detections': len(df),
        'unique_classes': df['class_name'].nunique(),
        'avg_confidence': df['confidence'].mean(),
        'high_confidence_rate': (df['confidence'] > 0.5).mean(),
        'low_confidence_rate': (df['confidence'] < 0.3).sum() / len(df) if len(df) > 0 else 0,
        'class_distribution': class_counts.to_dict('records'),
        'confidence_stats': confidence_stats.to_dict(),
        'spatial_stats': spatial_stats
    }


def evaluate_model_performance(all_detection_data, inference_times):
    """Evaluate model performance across all test images"""
    if not all_detection_data or len(all_detection_data) == 0:
        print("No detection data available")
        return None

    # Combine all detections
    all_detections = []
    for img_name, detections in all_detection_data.items():
        for det in detections:
            det['image'] = img_name
            all_detections.append(det)

    # Create DataFrame
    df = pd.DataFrame(all_detections)

    # Overall statistics
    total_images = len(all_detection_data)
    total_detections = len(df)
    avg_detections_per_image = total_detections / total_images if total_images > 0 else 0

    # Confidence statistics
    avg_confidence = df['confidence'].mean()
    high_confidence_rate = (df['confidence'] > 0.5).mean()
    low_confidence_rate = (df['confidence'] < 0.3).sum() / len(df) if len(df) > 0 else 0

    # Class distribution
    class_counts = df['class_name'].value_counts()

    # Performance metrics
    avg_inference_time = np.mean(list(inference_times.values()))
    max_inference_time = max(inference_times.values())
    min_inference_time = min(inference_times.values())
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    # Results by image
    results_by_image = {}
    for img_name, img_detections in all_detection_data.items():
        detections_count = len(img_detections)
        avg_conf = np.mean([d['confidence'] for d in img_detections]) if img_detections else 0
        results_by_image[img_name] = {
            'detections': detections_count,
            'avg_confidence': avg_conf,
            'inference_time': inference_times.get(img_name, 0)
        }

    return {
        'summary': {
            'total_images': total_images,
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections_per_image,
            'avg_confidence': avg_confidence,
            'high_confidence_rate': high_confidence_rate,
            'low_confidence_rate': low_confidence_rate,
            'avg_inference_time': avg_inference_time,
            'max_inference_time': max_inference_time,
            'min_inference_time': min_inference_time,
            'fps': fps
        },
        'class_distribution': class_counts.to_dict(),
        'results_by_image': results_by_image
    }


def generate_performance_visualizations(performance, output_dir):
    """Generate visualizations for model performance"""
    # Create a directory for charts
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Class distribution chart (top 15 classes)
    plt.figure(figsize=(12, 8))
    class_items = list(performance['class_distribution'].items())
    top_classes = sorted(class_items, key=lambda x: x[1], reverse=True)[:15]

    classes = [c[0] for c in top_classes]
    counts = [c[1] for c in top_classes]

    plt.barh(classes, counts)
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.title('Top 15 Detected Music Symbol Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    # 2. Detection count by image
    plt.figure(figsize=(12, 8))
    images = list(performance['results_by_image'].keys())
    det_counts = [performance['results_by_image'][img]['detections'] for img in images]

    # Sort by detection count
    sorted_idx = np.argsort(det_counts)[::-1]
    images = [images[i] for i in sorted_idx]
    det_counts = [det_counts[i] for i in sorted_idx]

    # Limit to top 20 images if there are many
    if len(images) > 20:
        images = images[:20]
        det_counts = det_counts[:20]

    plt.bar(images, det_counts)
    plt.xlabel('Image')
    plt.ylabel('Detection Count')
    plt.title('Number of Detections by Image')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'detection_by_image.png'), dpi=300)
    plt.close()

    # 3. Inference time by image
    plt.figure(figsize=(12, 8))
    images = list(performance['results_by_image'].keys())
    inf_times = [performance['results_by_image'][img]['inference_time'] for img in images]

    # Sort by inference time
    sorted_idx = np.argsort(inf_times)[::-1]
    images = [images[i] for i in sorted_idx]
    inf_times = [inf_times[i] for i in sorted_idx]

    # Limit to top 20 images if there are many
    if len(images) > 20:
        images = images[:20]
        inf_times = inf_times[:20]

    plt.bar(images, inf_times)
    plt.xlabel('Image')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time by Image')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'inference_time.png'), dpi=300)
    plt.close()

    return charts_dir


def main():
    # Parse arguments
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    print(f"\n{'='*50}")
    print(f"OMR Model Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load model
    model = load_model(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Set image size
    model.conf = args.conf
    model.iou = 0.45  # Default IoU threshold
    model.max_det = 1000  # Maximum detections per image
    model.imgsz = args.img_size

    # Load class mapping
    class_mapping = load_class_mapping(args.class_map)
    if class_mapping is None:
        print("Failed to load class mapping. Using default class IDs.")

    # Find all test images
    if os.path.isdir(args.test_dir):
        image_files = glob.glob(os.path.join(args.test_dir, '*.png')) + \
                     glob.glob(os.path.join(args.test_dir, '*.jpg')) + \
                     glob.glob(os.path.join(args.test_dir, '*.jpeg'))
    else:
        # Single image file
        image_files = [args.test_dir] if os.path.exists(args.test_dir) else []

    if not image_files:
        print(f"No images found in {args.test_dir}")
        return

    print(f"Found {len(image_files)} test images")

    # Process each image
    all_detection_data = {}
    inference_times = {}

    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = os.path.basename(img_path)
        print(f"\nProcessing {img_name}")

        # Run inference
        detection_data, results, inf_time = run_inference(model, img_path, args.conf, class_mapping)
        inference_times[img_name] = inf_time

        if detection_data:
            # Store detection data
            all_detection_data[img_name] = detection_data

            # Create visualization
            vis_output_path = os.path.join(args.output_dir, 'visualizations', f"{os.path.splitext(img_name)[0]}_detection.png")
            visualize_detection_with_labels(img_path, detection_data, vis_output_path)

            # Analyze this image's detections
            analysis = analyze_detections(detection_data)

            # Print some statistics
            print(f"  - Detected {len(detection_data)} symbols in {inf_time:.4f} seconds")
            print(f"  - Average confidence: {analysis['avg_confidence']:.4f}")

            # Save JSON results if requested
            if args.save_json:
                json_output_path = os.path.join(args.output_dir, f"{os.path.splitext(img_name)[0]}_detections.json")
                with open(json_output_path, 'w') as f:
                    json.dump({
                        'image': img_name,
                        'detections': detection_data,
                        'analysis': analysis,
                        'inference_time': inf_time
                    }, f, indent=2)

    # Evaluate overall model performance
    print("\nEvaluating overall model performance...")
    performance = evaluate_model_performance(all_detection_data, inference_times)

    # Generate performance visualizations
    charts_dir = generate_performance_visualizations(performance, args.output_dir)

    # Save performance report
    report_path = os.path.join(args.output_dir, 'performance_report.json')
    with open(report_path, 'w') as f:
        json.dump(performance, f, indent=2)

    # Generate human-readable report
    txt_report_path = os.path.join(args.output_dir, 'performance_report.txt')
    with open(txt_report_path, 'w') as f:
        f.write("OMR Model Evaluation Report\n")
        f.write("=========================\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Confidence threshold: {args.conf}\n")
        f.write(f"Images evaluated: {performance['summary']['total_images']}\n\n")

        f.write("Overall Performance:\n")
        f.write(f"- Total detections: {performance['summary']['total_detections']}\n")
        f.write(f"- Average detections per image: {performance['summary']['avg_detections_per_image']:.2f}\n")
        f.write(f"- Average confidence: {performance['summary']['avg_confidence']:.4f}\n")
        f.write(f"- High confidence detection rate (>0.5): {performance['summary']['high_confidence_rate']:.4f}\n")
        f.write(f"- Low confidence detection rate (<0.3): {performance['summary']['low_confidence_rate']:.4f}\n\n")

        f.write("Inference Performance:\n")
        f.write(f"- Average inference time: {performance['summary']['avg_inference_time']:.4f} seconds\n")
        f.write(f"- FPS: {performance['summary']['fps']:.2f}\n")
        f.write(f"- Min inference time: {performance['summary']['min_inference_time']:.4f} seconds\n")
        f.write(f"- Max inference time: {performance['summary']['max_inference_time']:.4f} seconds\n\n")

        f.write("Class Distribution (top 10):\n")
        for i, (cls, count) in enumerate(sorted(performance['class_distribution'].items(),
                                              key=lambda x: x[1], reverse=True)[:10]):
            f.write(f"- {cls}: {count}\n")

        f.write("\n\nResults by Image (top 10):\n")
        for i, (img, stats) in enumerate(sorted(performance['results_by_image'].items(),
                                            key=lambda x: x[1]['detections'], reverse=True)[:10]):
            f.write(f"- {img}: {stats['detections']} detections, ")
            f.write(f"avg confidence: {stats['avg_confidence']:.4f}, ")
            f.write(f"inference time: {stats['inference_time']:.4f}s\n")

    print(f"\nEvaluation complete. Results saved to {args.output_dir}")
    print(f"Performance report: {report_path}")
    print(f"Visualizations: {os.path.join(args.output_dir, 'visualizations')}")
    print(f"Charts: {charts_dir}")
    print(f"\nSummary:")
    print(f"- Total detections: {performance['summary']['total_detections']}")
    print(f"- Average detections per image: {performance['summary']['avg_detections_per_image']:.2f}")
    print(f"- Average confidence: {performance['summary']['avg_confidence']:.4f}")
    print(f"- Average inference time: {performance['summary']['avg_inference_time']:.4f} seconds")
    print(f"- FPS: {performance['summary']['fps']:.2f}")


if __name__ == "__main__":
    main()
