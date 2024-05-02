from donkeycar.pipeline.types import TubDataset, TubRecord
import os
import donkeycar as dk
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import cv2
import re

# python3 /home/artur/mycar/open_loop_validation.py sunny_25k.h5 --shuffle

# Number of frames the model was trained on
MODEL_FRAMES_SIZES = {
        r"dark_2k": 2418,
        r"dark_25k*": 23243,
        r"cloudy_2k": 2587,
        r"cloudy_25k*": 24402,
        r"sunny_2k": 2410,
        r"sunny_25k": 24023,
        r"supermodel": 71668
    }

VALIDATION_TUBS = {
    r"sunny.*": ["tub_validation_sunny_boxes"],
    r"cloudy.*": ["tub_validation_cloudy_boxes"],
    r"dark.*": ["tub_validation_dark_boxes"],
    r"supermodel.*": ["tub_validation_sunny_boxes", "tub_validation_cloudy_boxes", "tub_validation_dark_boxes"],
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on validation data.")
    parser.add_argument("model", type=str, help="Name of the pre-trained model file (with file extension)")
    return parser.parse_args()

def get_training_frames_size(model_name):
    for pattern, size in MODEL_FRAMES_SIZES.items():
        if re.match(pattern, model_name):
            return size
    return None

def get_val_tub_names(model_name):
    for pattern, tubs in VALIDATION_TUBS.items():
        if re.match(pattern, model_name):
            return tubs
    return None

def load_records(tub_name, cfg, data_path):
    dataset = TubDataset(
        config=cfg,
        tub_paths=[os.path.expanduser(os.path.join(data_path, tub_name))])

    records = dataset.get_records()
    for record in records:
        record.underlying['cam/image_array'] = tub_name + "/images/" + record.underlying['cam/image_array']
    return records
    

def get_true_labels(record: TubRecord):
  return [r.underlying['user/angle'] for r in record]

def get_predictions(model, records, tub_path):
    predictions = []
    for record in records:
        rgb_image = cv2.imread(tub_path + record.underlying['cam/image_array'])
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # rgb to bgr
        image = image.astype(np.float64) * (1.0 / 255.0) # normalize the image
        image = np.expand_dims(image, axis=0)
        angle, throttle = model.predict(image)
        predictions.append(angle[0][0])
    return predictions

def mean_absolute_error(predictions, ground_truths):
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    return np.mean(np.abs(predictions - ground_truths))

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load config
    cfg = dk.load_config(config_path='./config.py')

    # validation tub path
    tub_path = "./data/"
    tub_names = get_val_tub_names(args.model)

    MAEs = {}
    for tub in tub_names:
        test_data = load_records(tub, cfg, tub_path)
        # Get true labels of each image
        y_test = get_true_labels(test_data)
        
        # Shuffle data
        x_test, _, y_test, _ = train_test_split(test_data, y_test, test_size=1, shuffle=True, random_state=args.random_state)

        # Get training frames
        frames = get_training_frames_size(args.model)
        test_data_size = None
        if frames != None:
            test_data_size = round((frames/8)) * 2 # 80% was train data, 20% will be test_data
        
        if len(test_data) > test_data_size:
            x_test = x_test[:test_data_size]
            y_test = y_test[:test_data_size]

        print("Train data size:", frames)
        print("Validation data size:", len(x_test))

        # Predict
        model = load_model(f"models/{args.model}")    
        predictions = get_predictions(model, x_test, tub_path)

        # Calculate mean absolute error
        mae = mean_absolute_error(predictions, y_test)
        MAEs[tub] = mae

    for tub, mae in MAEs.items():
        print("Validation data:", tub)
        print("Mean Absolute Error:", mae)
        print()

if __name__ == "__main__":
    main()