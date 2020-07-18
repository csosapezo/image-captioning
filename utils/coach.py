import logging
import os
import sys
from pickle import Unpickler

import numpy as np
import tensorflow
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from model.keras import ImageCaptioningModel
from utils import data_generator, constants

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, max_length, vocabulary_size, model=ImageCaptioningModel, generator=data_generator):
        self.generator = generator
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.model = model(max_length=max_length, vocabulary_size=vocabulary_size)

    @staticmethod
    def _load_features_dict(image_dict_path):
        if not os.path.isfile(image_dict_path):
            log.error(f"File {image_dict_path} not found.")
            sys.exit(constants.ERROR_EXIT)
        else:
            with open(image_dict_path, "rb") as f:
                features_dict = Unpickler(f).load()

            return features_dict

    def predict_word(self, feature_vector, tokenizer):
        cur_seq = constants.START_SEQ
        predicted_caption = []
        partial_captions = np.zeros(self.max_length)
        partial_captions[0] = tokenizer.word_index[cur_seq]
        num_seqs = 1

        while num_seqs < self.max_length and cur_seq != constants.END_SEQ:
            # Next sequence returned on one-hot encode
            pred_seq_vector = self.model.predict(np.array([feature_vector]), np.array([partial_captions]))[0]
            # one-hot encode → integer
            pred_seq_id = tensorflow.math.argmax(pred_seq_vector).numpy()
            # integer → word
            pred_seq = tokenizer.index_word[pred_seq_id]
            predicted_caption.append(pred_seq)
            # add predicted seq to prediction
            cur_seq = pred_seq
            partial_captions[num_seqs] = pred_seq_id
            num_seqs += 1

        return predicted_caption

    def train(self, image_dict_path, captions_vector, epochs, batch_size, verbose):
        features_dict = self._load_features_dict(image_dict_path)
        return self.model.fit(generator=self.generator(captions_vector, features_dict,
                                                       batch_size, self.vocabulary_size, self.max_length),
                              epochs=epochs,
                              train_size=len(captions_vector),
                              batch_size=batch_size,
                              verbose=verbose)

    def validate(self, image_dict_path, captions_vector, tokenizer):
        features_dict = self._load_features_dict(image_dict_path)
        bleu_score = 0
        log.info(f"Validating {self.model.__class__.__name__} on {len(captions_vector)} samples...")
        for key, sequence in tqdm(captions_vector):
            feature_vector = features_dict[key]
            predicted_caption = self.predict_word(feature_vector, tokenizer)
            real_caption = tokenizer.sequences_to_texts([list(sequence)])[0].split()
            print(real_caption)
            input(predicted_caption)
            real_caption.pop(0)
            bleu_score += sentence_bleu([real_caption], predicted_caption)

        bleu_score /= len(captions_vector)
        log.info(f"Validation done! Average BLEU Score: {bleu_score:.4f}")

    def save_model(self, save_dir=constants.SAVE_DIR, filename=constants.FILENAME):
        filepath = os.path.join(save_dir, filename)
        log.info(f"Looking for directory {save_dir} if exists...")
        if not os.path.exists(save_dir):
            log.info(f"Directory {save_dir} not found. Creating directory...")
            os.mkdir(save_dir)
        log.info(f"Saving model...")
        filepath = self.model.save_weights(filepath)
        log.info(f"Model saved at {filepath}!")
        return filepath[filepath.rfind('/') + 1:]

    def load_model(self, load_dir=constants.SAVE_DIR, filename=None):
        filepath = os.path.join(load_dir, filename)
        if not filename:
            log.error("Saved model filename not given!")
            sys.exit(constants.ERROR_EXIT)
        elif not os.path.exists(load_dir):
            log.error(f"Directory {load_dir} not found!")
            sys.exit(constants.ERROR_EXIT)
        elif not os.path.exists(filepath):
            log.error(f"Model file {filename} not found at {load_dir}!")
            sys.exit(constants.ERROR_EXIT)
        else:
            self.model.load_weights(filepath)




