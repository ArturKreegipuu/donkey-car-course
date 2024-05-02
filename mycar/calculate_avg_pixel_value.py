from donkeycar.pipeline.types import TubDataset
import donkeycar as dk
import os
import numpy as np
import argparse
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate average pixel value of a data tub")
    parser.add_argument("tub", type=str, help="Name of the tub")
    return parser.parse_args()

def load_records(tub_name, cfg, data_path):
    dataset = TubDataset(
        config=cfg,
        tub_paths=[os.path.expanduser(os.path.join(data_path, tub_name))])

    records = dataset.get_records()
    return records

def get_avg_pixel_value(record, tub_path):
    rgb_image = cv2.imread(tub_path + record.underlying['cam/image_array'])
    # Calculate the average pixel value across all color channels
    avg_pixel_value = np.mean(rgb_image)
    return avg_pixel_value
# Main script
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load test data
    tub_name = args.tub
    data = load_records(tub_name, None, "./data/")
    total = 0
    for record in data:
        total += get_avg_pixel_value(record, "./data/" + tub_name + "/images/")
    avg_pixel_value_dataset = total / len(data)
    print("Average pixel value of the dataset:", avg_pixel_value_dataset)

if __name__ == "__main__":
    main()