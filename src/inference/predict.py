import os
import json
import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import yaml

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2].absolute()

def load_config(config_file=None):
    if not config_file:
        config_file = os.path.join(PROJECT_ROOT, "configs", "inference_config.yaml")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {
            "weights": os.path.join(PROJECT_ROOT, "output", "models", "best.pt"),
            "conf_threshold": 0.15,
            "class_mapping": os.path.join(PROJECT_ROOT, "raw_data", "class_mapping.csv"),
            "output_dir": os.path.join(PROJECT_ROOT, "output", "results"),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu"
        }

def load_model(weights_file=None):
    config = load_config()

    # Use default weights if none provided
    if not weights_file:
        weights_file = config["weights"]
        if not os.path.isabs(weights_file):
            weights_file = os.path.join(PROJECT_ROOT, weights_file)

    # Check if weights file exists
    if not os.path.exists(weights_file):
        print(f"Can't find the model weights at {weights_file}")
        return None

    # Try to load the model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file)
        print(f"Loaded the model from {weights_file}")
        return model
    except Exception as e:
        print(f"Couldn't load the model: {e}")
        return None

def load_class_mapping(mapping_file=None):
    config = load_config()

    # Use default mapping file if none given
    if not mapping_file:
        mapping_file = config["class_mapping"]
        if not os.path.isabs(mapping_file):
            mapping_file = os.path.join(PROJECT_ROOT, mapping_file)

    # Check if mapping file exists
    if not os.path.exists(mapping_file):
        print(f"Can't find the class mapping at {mapping_file}")
        return None

    # Try to load the mapping
    try:
        classes = pd.read_csv(mapping_file)

        # Make dictionaries to convert between different class ids
        id_to_name = dict(zip(classes['original_id'], classes['class_name']))
        yolo_to_id = dict(zip(classes['yolo_id'], classes['original_id']))
        yolo_to_name = dict(zip(classes['yolo_id'], classes['class_name']))

        return {
            'id_to_name': id_to_name,
            'yolo_to_id': yolo_to_id,
            'yolo_to_name': yolo_to_name
        }
    except Exception as e:
        print(f"Couldn't load the class mapping: {e}")
        return None

def run_inference(model, image_file, conf_threshold=None, class_mapping=None):
    """Run model inference on an image"""
    config = load_config()

    # Get confidence threshold from config if not provided
    if conf_threshold is None:
        conf_threshold = config["conf_threshold"]

    # Check if we have everything we need
    if not model:
        print("No model loaded")
        return None, None

    if not os.path.exists(image_file):
        print(f"Can't find the image at {image_file}")
        return None, None

    print(f"Running model on {image_file} (confidence > {conf_threshold})")

    # Set confidence threshold
    model.conf = conf_threshold

    # Run the model
    results = model(image_file)

    # Get image size
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: Could not read image {image_file}")
        return None, None

    height, width = img.shape[:2]

    # Save all the detections
    detections = []

    # Look at each thing the model found
    for pred in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, class_id = pred

        # Figure out what class this is
        class_name = f"class_{int(class_id)}"
        original_id = int(class_id)

        if class_mapping and 'yolo_to_id' in class_mapping and 'yolo_to_name' in class_mapping:
            if int(class_id) in class_mapping['yolo_to_id']:
                original_id = class_mapping['yolo_to_id'][int(class_id)]

            if int(class_id) in class_mapping['yolo_to_name']:
                class_name = class_mapping['yolo_to_name'][int(class_id)]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Save all the info about this detection
        detections.append({
            'class_id': original_id,
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

    # Sort from left to right
    detections.sort(key=lambda x: x['x_center'])

    print(f"Found {len(detections)} music symbols")

    return detections, results

def predict_and_save(image_file, output_file=None, weights_file=None, conf_threshold=None):
    """Run prediction and save the results"""
    config = load_config()

    model = load_model(weights_file)
    if not model:
        return None


    class_mapping = load_class_mapping()


    detections, results = run_inference(model, image_file, conf_threshold, class_mapping)
    if not detections:
        return None

    if not output_file:
        output_dir = config["output_dir"]
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_detections.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"Saved results to {output_file}")
    return detections

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find music symbols in an image")
    parser.add_argument("image", help="image to look at")
    parser.add_argument("--output", "-o", help="where to save results")
    parser.add_argument("--weights", "-w", help="model weights to use")
    parser.add_argument("--conf", "-c", type=float, help="how sure model needs to be (0-1)")
    parser.add_argument("--config", help="path to config file")

    args = parser.parse_args()

    if args.config:
        load_config(args.config)

    predict_and_save(args.image, args.output, args.weights, args.conf)
