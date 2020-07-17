import logging
import os
import sys
import time
from pickle import Pickler, Unpickler

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import text

from utils import constants

log = logging.getLogger(__name__)


class CaptionTokenizer:
    def __init__(self, top_keys=constants.TOP_KEYS):
        self.tokenizer = text.Tokenizer(num_words=top_keys, oov_token=constants.UNKNOWN, filters=constants.FILTERS)

    @staticmethod
    def _max_lenght(captions):
        return max(len(c) for c in captions)

    def _fit_transform(self, captions):
        self.tokenizer.fit_on_texts(captions)
        train_seqs = self.tokenizer.texts_to_sequences(captions)
        return train_seqs

    def _pad_sequences(self, train_seqs):
        self.tokenizer.word_index[constants.PAD] = 0
        self.tokenizer.index_word[0] = constants.PAD
        captions_vector = preprocessing.sequence.pad_sequences(train_seqs, padding='post')
        return captions_vector

    def _save_data(self, captions_vector, dump_dir, filenames):
        vector_filename = os.path.join(dump_dir, filenames[1])
        tokenizer_filename = os.path.join(dump_dir, filenames[0])
        # Save vector
        with open(vector_filename, "wb+") as cvf:
            Pickler(cvf).dump(captions_vector)
        # Save tokenizer
        with open(tokenizer_filename, "wb+") as tkf:
            Pickler(tkf).dump(self.tokenizer)

        return vector_filename, tokenizer_filename

    def load_data(self, dump_dir, filename):
        tokenizer_filename = os.path.join(dump_dir, filename)
        if not os.path.isfile(tokenizer_filename):
            log.error(f"File {tokenizer_filename} not found.")
            sys.exit(constants.ERROR_EXIT)
        else:
            with open(tokenizer_filename, "rb") as f:
                self.tokenizer = Unpickler(f).load()

    def tokenize(self, captions, dump_dir=constants.TEMP_DIR,
                 filenames=(constants.TOKENIZER_FILE_NAME, constants.CAPTIONS_VECTOR_FILE_NAME)):
        log.info("Tokenizing image captions...")
        start = time.time()
        train_seqs = self._fit_transform(captions=captions)
        captions_vector = self._pad_sequences(train_seqs=train_seqs)
        max_lenght = self._max_lenght(captions=captions_vector)
        end = time.time()
        log.info(f"Caption vectors obtained! Total elapsed time: {(end - start):.2f} s")
        vector_filename, tokenizer_filename = self._save_data(captions_vector, dump_dir, filenames)
        log.info(f"Captions vectors saved at {vector_filename}!")
        log.info(f"Tokenizer vectors saved at {tokenizer_filename}!")
        return max_lenght, captions_vector




