import json
import logging
import time

from sklearn.utils import shuffle
from tqdm import tqdm

from utils import constants

log = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, data_type, annotation_file, data_dir, limit=None, random_state=constants.RANDOM_STATE):
        # Variables to load data
        self.data_type = data_type
        self.annotation_file = annotation_file
        self.data_dir = data_dir
        self.limit = limit
        self.random_state = random_state

        # Temporal variables
        self.annotations = None

        # output variables
        self.captions = []
        self.image_paths = []

    def _open_annotation_file(self):
        log.info(f"Searching annotation file on path {self.annotation_file}...")
        with open(self.annotation_file, 'r') as file:
            log.info("File found! Reading annotations from file...")

            start = time.time()
            self.annotations = json.load(file).get("annotations")
            end = time.time()

            log.info(f"Annotations read! Total elapsed time: {(end - start):.2f} s")

    def _load_captions(self):
        log.info("Loading captions from annotation files...")
        for ann in tqdm(self.annotations):
            caption = constants.START_SEQ + constants.BLANK + ann['caption'] + constants.BLANK + constants.END_SEQ
            image_id = ann['image_id']
            image_path = constants.IMAGE_PATH.format(images_dir=self.data_dir,
                                                     data_type=self.data_type,
                                                     image_id=str(image_id).zfill(constants.ID_SIZE))
            self.captions.append(caption)
            self.image_paths.append(image_path)

    def _shuffle(self):
        log.info("Shuffling the data...")
        start = time.time()
        self.captions, self.image_paths = shuffle(self.captions, self.image_paths, random_state=self.random_state)
        end = time.time()
        log.info(f"Data shuffled! Total elapsed time: {(end - start):.2f} s")

    def _sample(self):
        if self.limit:
            log.info(f"Selecting {self.limit} samples from dataset...")
            self.captions = self.captions[:self.limit]
            self.image_paths = self.image_paths[:self.limit]

    def load(self):
        start = time.time()
        self._open_annotation_file()
        self._load_captions()
        self._shuffle()
        self._sample()
        end = time.time()
        log.info(f"Data loaded! Total elapsed time: {(end - start):.2f} s")
        return self.captions, self.image_paths
