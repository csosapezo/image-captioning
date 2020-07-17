import os
from pickle import Pickler

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3
from tqdm import tqdm

from utils import constants, ImageLoader


class FeatureExtractor:
    def __init__(self, base_image_mode=InceptionV3, preprocess_input=inception_v3.preprocess_input,
                 image_paths=None, dim=constants.INCEPTION_DIM):
        self.model = base_image_mode(weights=constants.IMAGE_WEIGHTS)
        self.model = Model(self.model.input, self.model.layers[-2].output)
        self.image_paths = image_paths or []
        self.image_loader = ImageLoader(dim=dim, preprocess_input=preprocess_input)

    @staticmethod
    def _save_feature_dict(dump_dir, feature_dict):
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        filename = os.path.join(dump_dir, constants.FEATURE_DICT_NAME)
        with open(filename, "wb+") as f:
            Pickler(f).dump(feature_dict)

    def load_and_preprocess_one(self, path):
        img = self.image_loader.load(path)
        feature = self.model.predict(img)
        feature = np.reshape(feature, feature.shape[1])

        return feature

    def __call__(self, dump_dir=constants.TEMP_DIR):
        unique_image_paths = sorted(set(self.image_paths))
        feature_dict = {}

        for path in tqdm(unique_image_paths):
            feature = self.load_and_preprocess_one(path)
            feature_dict[path] = feature

        self._save_feature_dict(dump_dir=dump_dir, feature_dict=feature_dict)
