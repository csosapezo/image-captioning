from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout, Dense, Embedding, add
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam

from utils import constants


class ImageCaptioningModel:
    def __init__(self, max_length, vocabulary_size, dropout=constants.DROPOUT, embedding_dim=constants.EMBEDDING_DIM,
                 input_size=constants.INCEPTION_OUTPUT, learning_rate=constants.LEARNING_RATE):

        # Training variables
        self.epochs = None

        # image feature extraction
        image_features = Input(shape=(input_size,))
        feature_extraction_hidden = Dropout(dropout)(image_features)
        feature_extraction_output = Dense(embedding_dim, activation='relu')(feature_extraction_hidden)

        # partial caption sequence
        partial_captions = Input(shape=(max_length,))
        embedding_layer = Embedding(vocabulary_size, embedding_dim, mask_zero=True)(partial_captions)
        sequence_dropout = Dropout(dropout)(embedding_layer)
        sequence_output = GRU(units=embedding_dim, recurrent_initializer='glorot_uniform')(sequence_dropout)

        # decoder
        decoder_input = add([feature_extraction_output, sequence_output])
        decoder_hidden = Dense(embedding_dim, activation='relu')(decoder_input)
        outputs = Dense(vocabulary_size, activation='softmax')(decoder_hidden)
        # full model
        self.model = Model(inputs=[image_features, partial_captions], outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

    def fit(self, generator, epochs, train_size, batch_size, verbose):
        self.epochs = epochs
        return self.model.fit(x=generator,
                              steps_per_epoch=train_size // batch_size,
                              epochs=epochs,
                              verbose=verbose)

    def predict(self, image, caption):
        return self.model.predict(x=[image, caption])

    def save_weights(self, filename):
        self.model.save_weights(filename.format(epochs=self.epochs, name=ImageCaptioningModel.__name__))
        return filename.format(epochs=self.epochs, name=ImageCaptioningModel.__name__)

    def load_weights(self, filename):
        self.model.load_weights(filename)
