import os
import yaml
import pandas as pd
import subprocess
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2].absolute()

def load_config(config_file=None):
    """Load the training configuration"""
    if not config_file:
        config_file = os.path.join(PROJECT_ROOT, "configs", "training_config.yaml")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded the config from {config_file}")
            return config
    except Exception as e:
        print(f"Couldn't load the config file: {e}")
        return None

def make_data_yaml(csv_file, yaml_file, train_folder, val_folder, train_labels, val_labels):
    """Create the data.yaml file that YOLOv5 needs for training"""
    # Make paths absolute if they're not
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(PROJECT_ROOT, csv_file)

    # Read the class mapping CSV
    try:
        classes = pd.read_csv(csv_file)
        print(f"Found {len(classes)} different classes to detect")
    except Exception as e:
        print(f"Error reading class mapping file: {e}")
        return False

    # Make a list of class names
    if 'class_name' in classes.columns:
        class_names = classes.sort_values('original_id')['class_name'].tolist()
    else:
        class_names = [f"class_{i}" for i in range(len(classes))]

    # Make paths absolute if they're not
    if not os.path.isabs(train_folder):
        train_folder = os.path.join(PROJECT_ROOT, train_folder)
    if not os.path.isabs(val_folder):
        val_folder = os.path.join(PROJECT_ROOT, val_folder)
    if not os.path.isabs(train_labels):
        train_labels = os.path.join(PROJECT_ROOT, train_labels)
    if not os.path.isabs(val_labels):
        val_labels = os.path.join(PROJECT_ROOT, val_labels)

    # Make the yaml content
    yaml_content = {
        'path': str(PROJECT_ROOT),
        'train': train_folder,
        'val': val_folder,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'nc': len(classes),
        'names': class_names
    }

    # Save it to a file
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)

        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        print(f"Made the YAML file at {yaml_file}")
        return True
    except Exception as e:
        print(f"Couldn't write the YAML file: {e}")
        return False

def train_model(config_file=None):
    """Train the YOLOv5 model with the specified configuration"""
    # Load configuration
    config = load_config(config_file)
    if not config:
        return None

    # Get all the settings from config
    model_type = config.get('model', 'yolov5s')
    weights = config.get('weights', f"{model_type}.pt")
    img_size = config.get('img_size', 1024)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 100)
    train_path = config.get('train_path', "raw_data/images/train")
    val_path = config.get('val_path', "raw_data/images/val")
    train_labels = config.get('train_labels', "raw_data/labels/train")
    val_labels = config.get('val_labels', "raw_data/labels/val")
    data_yaml = config.get('data_yaml', "data.yaml")
    class_mapping = config.get('class_mapping', "raw_data/class_mapping.csv")
    output_dir = config.get('output_dir', "output")
    device = config.get('device', 'cuda:0')

    # Make paths absolute if they're not
    if not os.path.isabs(data_yaml):
        data_yaml = os.path.join(PROJECT_ROOT, data_yaml)

    # Make output folder if it doesn't exist
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check for models directory and create it
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Check if we can use GPU
    has_gpu = subprocess.run('nvidia-smi', shell=True, capture_output=True).returncode == 0
    if not has_gpu and device.startswith('cuda'):
        print("No GPU found, using CPU instead")
        device = 'cpu'

    # Get YOLOv5 if we don't have it
    yolo_dir = os.path.join(PROJECT_ROOT, 'yolov5')
    if not os.path.exists(yolo_dir):
        print("Getting YOLOv5...")
        os.system(f'git clone https://github.com/ultralytics/yolov5.git {yolo_dir}')
        os.system(f'pip install -r {yolo_dir}/requirements.txt')

    # Check if we have the class mapping file
    if not os.path.isabs(class_mapping):
        class_mapping = os.path.join(PROJECT_ROOT, class_mapping)

    if not os.path.exists(class_mapping):
        print(f"Can't find the class mapping file at {class_mapping}")
        return None

    # Make data.yaml file
    print(f"Making {data_yaml} from {class_mapping}...")
    success = make_data_yaml(
        class_mapping,
        data_yaml,
        train_path,
        val_path,
        train_labels,
        val_labels
    )

    if not success:
        print(f"Couldn't make {data_yaml}, stopping training")
        return None

    # Make the training command
    train_script = os.path.join(yolo_dir, 'train.py')
    command = f"python {train_script} --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights {weights} --device {device} --name omr_model --exist-ok"

    print(f"Starting training with command: {command}")

    # Run the training
    os.system(command)

    # Copy the trained model to output folder
    exp_dir = os.path.join(yolo_dir, 'runs', 'train')
    if os.path.exists(exp_dir):
        # Get the latest training run
        latest_run = sorted(os.listdir(exp_dir))[-1]
        weights_dir = os.path.join(exp_dir, latest_run, 'weights')

        if os.path.exists(weights_dir):
            # Copy best and last weights
            best_weights = os.path.join(weights_dir, 'best.pt')
            last_weights = os.path.join(weights_dir, 'last.pt')

            if os.path.exists(best_weights):
                os.system(f"cp {best_weights} {models_dir}/")
                print(f"Saved best model to {models_dir}/best.pt")

            if os.path.exists(last_weights):
                os.system(f"cp {last_weights} {models_dir}/")
                print(f"Saved last model to {models_dir}/last.pt")

    print("All done!")
    return os.path.join(models_dir, "best.pt")

if __name__ == "__main__":
    import argparse

    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv5 for sheet music')
    parser.add_argument('--config', type=str,
                      help='config file to use')

    # Add options to override config settings
    parser.add_argument('--img-size', type=int, help='change image size')
    parser.add_argument('--batch-size', type=int, help='change batch size')
    parser.add_argument('--epochs', type=int, help='change number of epochs')
    parser.add_argument('--weights', type=str, help='change initial weights')
    parser.add_argument('--device', type=str, help='change device')

    args = parser.parse_args()

    # Use provided config or default
    config_file = args.config

    # Load config
    config = load_config(config_file)
    if not config:
        print("Using default configuration")
        config = {}

    # Override config with command line args if given
    if args.img_size:
        config['img_size'] = args.img_size
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.weights:
        config['weights'] = args.weights
    if args.device:
        config['device'] = args.device

    # Save updated config if we loaded one from file
    if config_file:
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

    # Train the model
    train_model(config_file)
