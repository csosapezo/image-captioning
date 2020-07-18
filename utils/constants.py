# Utils
RANDOM_STATE = 0
ERROR_EXIT = 1
BLANK = ' '

# Captions loading
START_SEQ = '<start>'
END_SEQ = '<end>'

# Image loading and preprocessing
IMAGE_PATH = '{images_dir}/image/COCO_{data_type}_{image_id}.jpg'
ID_SIZE = 12
INCEPTION_DIM = (299, 299)
INCEPTION_OUTPUT = 2048
LIMIT = 10000

# Image model
IMAGE_WEIGHTS = 'imagenet'
TEMP_DIR = 'temp/'
FEATURE_DICT_NAME = 'feature_dict'

# Tokenization
TOP_KEYS = 6000
FILTERS = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
UNKNOWN = '<unknown>'
PAD = '<pad>'
CAPTIONS_VECTOR_FILE_NAME = 'captions_vector'
TOKENIZER_FILE_NAME = 'tokenizer'


# Captioning preprocessing
TEST_SIZE = 0.1

# Image captioning model
DROPOUT = 0.5
EMBEDDING_DIM = 256
LEARNING_RATE = 0.015
EPOCHS = 100
BATCH_SIZE = 128
VERBOSE = True
SAVE_DIR = 'model/weights'
FILENAME = 'model_{epochs}epochs_{name}'


# Extensions
MODEL_EXT = '.h5'
HISTORY_EXT = '.history'
