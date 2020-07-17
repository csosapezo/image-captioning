import argparse
import logging
import os
import time

import coloredlogs
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3
import matplotlib.pyplot as plt

from config import Environment
from utils import DataLoader, constants, CaptionTokenizer, FeatureExtractor, Coach

# log configuration
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def args_parser():
    parser.add_argument('-v', '--validate', dest='validate', action='store_const',
                        const=True, default=False, help='execute validation')
    parser.add_argument('model_path', nargs='?', help='model weights path')


def train(model_path, validation=False, base_image_mode=InceptionV3):
    # Load environment variables
    e = Environment()

    # Features dictionary file if exists
    features_dict_path = os.path.join(constants.TEMP_DIR, constants.FEATURE_DICT_NAME)

    # Start time

    # initialize data loader
    data_loader = DataLoader(data_type=e.data_type, annotation_file=e.captions_file,
                             data_dir=e.data_dir, limit=constants.LIMIT)

    # Loading and preprocessing data
    captions, image_paths = data_loader.load()

    # Extract image features
    log.info(f"Looking for already saved feature dictionary on file {features_dict_path}...")
    if not os.path.exists(features_dict_path):
        log.info(f"File {features_dict_path} not found. Starting feature extraction...")
        extractor = FeatureExtractor(image_paths=image_paths, base_image_mode=base_image_mode)
        extractor()
    else:
        log.info(f"File {features_dict_path} found!")

    # Tokenize image captions
    tokenizer = CaptionTokenizer()
    max_length, captions_vector = tokenizer.tokenize(captions=captions)
    captions_vector = list(zip(image_paths, captions_vector))

    # Create train and validation sets
    log.info(f"Splitting the dataset with a test size of {constants.TEST_SIZE:.2f}...")
    captions_train, captions_val = train_test_split(captions_vector,
                                                    test_size=constants.TEST_SIZE,
                                                    random_state=constants.RANDOM_STATE)

    # Train
    coach = Coach(max_length=max_length, vocabulary_size=constants.TOP_KEYS)

    if not validation:
        log.info(f"Training {coach.model.__class__.__name__} on {len(captions_train)} training samples...")
        train_history = coach.train(image_dict_path=features_dict_path,
                                    captions_vector=captions_train,
                                    epochs=constants.EPOCHS,
                                    batch_size=constants.BATCH_SIZE,
                                    verbose=constants.VERBOSE)

        model_filepath = coach.save_model(save_dir=constants.SAVE_DIR,
                                          filename=constants.FILENAME + constants.MODEL_EXT)

        log.info(f"Model saved at {model_filepath}!")
        log.info("Plotting training history...")

        # Plot training data
        plt.plot(train_history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

    else:
        coach.load_model(load_dir=constants.SAVE_DIR, filename=model_path)
        coach.validate(image_dict_path=features_dict_path,
                       captions_vector=captions_val,
                       tokenizer=tokenizer.tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and/or validate a image captioning model.')
    args_parser()
    args = parser.parse_args()
    filepath = None

    if args.validate:
        filepath = args.model_path
        log.info(f"Starting validation of model located at file {filepath}...")
    else:
        log.info("Starting training...")

    start = time.time()
    train(validation=args.validate, model_path=filepath)
    end = time.time()

    log.info(f"Process done! Total elapsed time: {(end - start):.2f} s")
