import argparse
import logging
import time

from utils import FeatureExtractor, CaptionTokenizer, constants, Coach

log = logging.getLogger(__name__)


def args_parser():
    parser.add_argument('image_path', nargs='?', help='image path')


def main(image_path, max_length=constants.MAX_LENGTH):
    # Load image and preprocess
    feature_extractor = FeatureExtractor()
    feature_vector = feature_extractor.load_and_preprocess_one(image_path)

    # Load existing tokenizer
    tokenizer = CaptionTokenizer()
    tokenizer.load_data(constants.TEMP_DIR, constants.TOKENIZER_FILE_NAME)

    # Make prediction
    coach = Coach(max_length=max_length, vocabulary_size=constants.TOP_KEYS)
    predicted_caption = coach.predict_word(feature_vector, tokenizer.tokenizer)
    constants.BLANK.join(predicted_caption)

    log.info(f"Generated caption for image: {predicted_caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a caption from an image.')
    args_parser()
    args = parser.parse_args()
    filepath = args.image_path

    log.info(f"Starting prediction for image {filepath}...")
    start = time.time()
    main(image_path=filepath)
    end = time.time()

    log.info(f"Process done! Total elapsed time: {(end - start):.2f} s")

