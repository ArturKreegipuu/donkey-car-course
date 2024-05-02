# donkey-car-course
Code changes made in the Donkey Car software for developing study materials utilizing the Donkey Car platform.



## Changes made to the Donkey Car source code:
---
* Keras Linear neural network architecture with an added resizing layer that triples the image's width and height: [donkeycar/donkeycar/parts/keras.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/parts/keras.py#L962)
---
* New class to enable the use of a slow model architecture: [donkeycar/donkeycar/parts/keras.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/parts/keras.py#L388)
---
* Keras Linear neural network architecture with an added cropping layer that crops 50 pixels off the top of an input image: [donkeycar/donkeycar/parts/keras.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/parts/keras.py#L927)
 ---
* New class to enable the use of a cropped model architecture: [donkeycar/donkeycar/parts/keras.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/parts/keras.py#L354)
---
* New command-line parameters to use custom neural networks: [donkeycar/donkeycar/utils.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/utils.py#L518)
  * Code changes allow to use `--type cropped` or `--type slow` to use a specific neural network architecture.
---
* Function to modify image labels to create "garbage" training data: [donkeycar/donkeycar/pipeline/training.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/pipeline/training.py#L128)
  * To enable this method `dataset.get_records()` needs to be changed to `modify_labels(dataset.get_records())` on [line 174](https://github.com/ArturKreegipuu/donkey-car-course/blob/3eb4740e587eb8b418da7e9e0321f7af28ed281e/donkeycar/donkeycar/pipeline/training.py#L174)
---

## [mycar](https://github.com/ArturKreegipuu/donkey-car-course/tree/main/mycar) folder
* [calculate_avg_pixel_value.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/main/mycar/calculate_avg_pixel_value.py)
  * Calculates the average pixel value of a data tub.
  * `python3 calculate_avg_pixel_value.py <tub_name>`
* [open_loop_validation.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/main/mycar/open_loop_validation.py)
  * Calculates the model's MAE by comparing the predicted values to the validation dataset's true labels.
  * `python3 open_loop_validation.py <model_name>`
* [myconfig.py](https://github.com/ArturKreegipuu/donkey-car-course/blob/main/mycar/myconfig.py)
  * Birghtness augmentation 
