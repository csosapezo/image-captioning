import os
from pickle import Pickler

from keras import Model
from keras.applications import InceptionV3
from tqdm import tqdm

from utils import constants, ImageLoader


class FeatureExtractor:
    def __init__(self, base_image_mode=InceptionV3, image_paths=None, dim=constants.INCEPTION_DIM):
        self.model = base_image_mode(weights=constants.IMAGE_WEIGHTS)
        self.model = Model(self.model.input, self.model.layers[-2].output)
        self.image_paths = image_paths or []
        self.image_loader = ImageLoader(dim=dim, model=base_image_mode)

    @staticmethod
    def __save_feature_dict(dump_dir, feature_dict):
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        filename = os.path.join(dump_dir, constants.FEATURE_DICT_NAME)
        with open(filename, "wb+") as f:
            Pickler(f).dump(feature_dict)

    def __call__(self, dump_dir=constants.TEMP_DIR):
        unique_image_paths = sorted(set(self.image_paths))
        feature_dict = {}

        for path in tqdm(unique_image_paths):
            img = self.image_loader.load(path)
            feature = self.model.predict(img)
            feature_dict[path] = feature
            self.__save_feature_dict(dump_dir=dump_dir, feature_dict=feature_dict)