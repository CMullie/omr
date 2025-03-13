import os
import yaml
import glob
import argparse
import subprocess
import pandas as pd
from pathlib import Path

def make_yaml(csv_file, yaml_file, train_img, val_img):
    # Read the CSV with classes
    try:
        classes = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    # Make sure we have the needed columns
    if 'original_id' not in classes.columns or 'yolo_id' not in classes.columns:
        print("CSV missing required columns")
        return False

    num_classes = len(classes)

    # Get class names
    if 'class_name' in classes.columns:
        classes = classes.sort_values(by='original_id')
        class_names = []
        for i, row in classes.iterrows():
            class_names.append(row['class_name'])
    else:
        class_names = []
        for i in range(num_classes):
            class_names.append(f"class_{i}")

    # Get paths
    current_dir = os.path.abspath(os.path.curdir)

    if not os.path.isabs(train_img):
        train_path_full = os.path.join(current_dir, train_img)
    else:
        train_path_full = train_img

    if not os.path.isabs(val_img):
        val_path_full = os.path.join(current_dir, val_img)
    else:
        val_path_full = val_img

    train_labels_full = os.path.join(current_dir, "labels/train")
    val_labels_full = os.path.join(current_dir, "labels/val")

    # Create YAML content
    yaml_data = {
        'path': current_dir,
        'train': train_path_full,
        'val': val_path_full,
        'train_labels': train_labels_full,
        'val_labels': val_labels_full,
        'nc': num_classes,
        'names': class_names
    }

    try:
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error writing YAML: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 model for OMR on GCP')
    parser.add_argument('--img-size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Initial weights')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Check GPU
    has_gpu = subprocess.run('nvidia-smi', shell=True, capture_output=True).returncode == 0
    device = 'cuda:0' if has_gpu else 'cpu'

    # Clone YOLOv5 if needed
    if not os.path.exists('yolov5'):
        os.system('git clone https://github.com/ultralytics/yolov5.git')
        os.system('pip install -r yolov5/requirements.txt')

    csv_path = 'class_mapping.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Can't find {csv_path}")
        return

    data_yaml_path = 'data.yaml'

    # Count label files
    train_files = glob.glob('labels/train/*.txt')
    val_files = glob.glob('labels/val/*.txt')

    # Make the YAML file
    success = make_yaml(
        csv_path,
        data_yaml_path,
        "./images/train",
        "./images/val"
    )

    if not success:
        print("Failed to create YAML file")
        return

    full_yaml_path = os.path.abspath(data_yaml_path)

    # Training command
    train_script = os.path.join('yolov5', 'train.py')
    command = (
        f"python3 {train_script} "
        f"--img {args.img_size} "
        f"--batch {args.batch_size} "
        f"--epochs {args.epochs} "
        f"--data {full_yaml_path} "
        f"--weights {args.weights} "
        f"--device {device} "
        f"--name omr_model "
        f"--exist-ok"
    )

    print(f"Running: {command}")

    # Run training
    os.system(command)

    # Copy results
    exp_folder = os.path.join('yolov5', 'runs', 'train')
    if os.path.exists(exp_folder):
        all_runs = os.listdir(exp_folder)
        if all_runs:
            latest_run = sorted(all_runs)[-1]
            weights_folder = os.path.join(exp_folder, latest_run, 'weights')

            if os.path.exists(weights_folder):
                best_model = os.path.join(weights_folder, 'best.pt')
                if os.path.exists(best_model):
                    os.system(f"cp {best_model} {args.output_dir}/")

                last_model = os.path.join(weights_folder, 'last.pt')
                if os.path.exists(last_model):
                    os.system(f"cp {last_model} {args.output_dir}/last.pt")

    print("Training completed")

if __name__ == "__main__":
    main()
