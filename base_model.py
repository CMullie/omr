import os
import numpy as np
import pandas as pd
import cv2
import torch
import yaml
import time
from datetime import datetime
import gc

# my settings
CONFIG = {
    'output_dir': './output',
    'models_dir': './models',
    'train_images_dir': './images/train',
    'val_images_dir': './images/val',
    'train_labels_dir': './labels/train',
    'val_labels_dir': './labels/val',
    'img_size': 640,
    'batch_size': 24,
    'epochs': 100,
    'pretrained_weights': 'yolov8n.pt',
    'confidence_threshold': 0.25,
    'iou_threshold': 0.35,
    'class_mapping_path': './class_mapping.csv',
}

os.makedirs('./output', exist_ok=True)
os.makedirs('./models', exist_ok=True)


class SheetMusicDetector:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.class_mapping = self.load_class_mapping(self.config['class_mapping_path'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.experiment_name = f"sheet_music_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def load_class_mapping(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} classes")
        original_id_mapping = dict(zip(df['original_id'], df['class_name']))
        yolo_id_mapping = dict(zip(df['yolo_id'], df['class_name']))
        return {'original_id': original_id_mapping, 'yolo_id': yolo_id_mapping}

    def load_model(self, weights_path=None):
        from ultralytics import YOLO

        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"Model loaded from {weights_path}")
        else:
            self.model = YOLO(self.config['pretrained_weights'])
            print(f"Using pre-trained model: {self.config['pretrained_weights']}")

    def prepare_dataset(self):
        train_img_dir = self.config['train_images_dir']
        val_img_dir = self.config['val_images_dir']
        train_label_dir = self.config['train_labels_dir']
        val_label_dir = self.config['val_labels_dir']

        train_images = len([f for f in os.listdir(train_img_dir)
                          if f.lower().endswith(('.png'))])
        val_images = len([f for f in os.listdir(val_img_dir)
                        if f.lower().endswith(('.png'))])

        train_labels = len([f for f in os.listdir(train_label_dir) if f.lower().endswith('.txt')])
        val_labels = len([f for f in os.listdir(val_label_dir) if f.lower().endswith('.txt')])

        print(f"Found {train_images} training images, {val_images} validation images")

        yaml_path = './dataset.yaml'
        self._create_yaml(yaml_path)

        return yaml_path

    def _create_yaml(self, yaml_path):
        max_id = max(int(yolo_id) for yolo_id in self.class_mapping['yolo_id'].keys())

        class_names = ['unknown'] * (max_id + 1)
        for yolo_id, class_name in self.class_mapping['yolo_id'].items():
            class_names[int(yolo_id)] = class_name

        config = {
            'path': './',
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return yaml_path

    def train(self, yaml_path, weights_path=None):
        from ultralytics import YOLO

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.model is None:
            self.load_model(weights_path or self.config['pretrained_weights'])

        output_dir = os.path.join(self.config['models_dir'], self.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            workers = min(8, os.cpu_count() or 1)

            self.model.train(
                data=yaml_path,
                epochs=self.config['epochs'],
                imgsz=self.config['img_size'],
                batch=self.config['batch_size'],
                name=self.experiment_name,
                project=self.config['models_dir'],
                patience=10,
                device=self.device,
                amp=True,
                workers=workers,
                save_period=10,
                exist_ok=True,
                pretrained=True,
                optimizer="AdamW",
                cos_lr=True
            )

            best_weights = os.path.join(self.config['models_dir'], self.experiment_name, 'weights', 'best.pt')

            if os.path.exists(best_weights):
                print(f"Model saved to {best_weights}")
                return best_weights
            else:
                last_weights = os.path.join(self.config['models_dir'], self.experiment_name, 'weights', 'last.pt')
                if os.path.exists(last_weights):
                    print(f"Using last weights: {last_weights}")
                    return last_weights
                else:
                    return None
        except Exception as e:
            print(f"Error: {e}")
            return None


def train_model():
    log_file = open(f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w")
    start_time = time.time()

    def log(txt):
        print(txt)
        log_file.write(txt + "\n")
        log_file.flush()

    log("="*40)
    log(f"Starting training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*40)

    try:
        # first training
        log("\nPhase 1: First training (20 epochs)")

        phase1_cfg = CONFIG.copy()
        phase1_cfg['img_size'] = 640
        phase1_cfg['batch_size'] = 16
        phase1_cfg['epochs'] = 20

        detector = SheetMusicDetector(phase1_cfg)
        yaml_path = detector.prepare_dataset()

        log("Training first model...")
        weights1 = detector.train(yaml_path)

        if not weights1:
            log("Training failed, trying with smaller batch...")
            phase1_cfg['batch_size'] = 8
            detector = SheetMusicDetector(phase1_cfg)
            weights1 = detector.train(yaml_path)

        # second training
        if weights1:
            log("\nPhase 2: Second training (80 epochs)")

            del detector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            phase2_cfg = CONFIG.copy()
            phase2_cfg['img_size'] = 800
            phase2_cfg['batch_size'] = 16
            phase2_cfg['epochs'] = 80
            phase2_cfg['pretrained_weights'] = weights1

            detector = SheetMusicDetector(phase2_cfg)

            log(f"Training second model...")
            weights2 = detector.train(yaml_path)

            if weights2:
                log(f"Training successful! Model: {weights2}")

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                gcs_path = f"gs://omr_simple/models/sheet_music_model_{timestamp}.pt"
                log(f"Copying to cloud: {gcs_path}")
                os.system(f"gsutil cp {weights2} {gcs_path}")

                return weights2
            else:
                log("Second training failed but we have the first model.")
                return weights1
        else:
            log("Training failed completely.")
            return None

    except Exception as e:
        log(f"Error: {e}")
        return None
    finally:
        total_mins = (time.time() - start_time) / 60
        log(f"\nTraining took {total_mins:.1f} minutes ({total_mins/60:.1f} hours)")
        log_file.close()


if __name__ == "__main__":
    print("Starting sheet music detector training...")
    weights = train_model()

    if weights:
        print(f"Success! Model saved to: {weights}")
    else:
        print("Training failed.")
