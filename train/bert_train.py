import os
import bert
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFBertForSequenceClassification

def create_tonkenizer(bert_layer):
    """Instantiate Tokenizer with vocab"""
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy() 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Vocab size:", len(tokenizer.vocab))
    return tokenizer

def get_ids(tokens, tokenizer, MAX_SEQ_LEN):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids

def get_masks(tokens, MAX_SEQ_LEN):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1] * len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))

def get_segments(tokens, MAX_SEQ_LEN):
    """Segments: 0 for the first sequence, 1 for the second"""  
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def create_single_input(sentence, tokenizer, max_len):
    """Create an input from a sentence"""
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:max_len] # max_len = MAX_SEQ_LEN - 2: reserved for [CLS] & [SEP]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    return get_ids(stokens, tokenizer, max_len+2), get_masks(stokens, max_len+2), get_segments(stokens, max_len+2)

def convert_sentences_to_features(sentences, tokenizer, MAX_SEQ_LEN):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences, position=0, leave=True):
      ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN-2) #-2: reserved for [CLS] & [SEP]
      
      input_ids.append(ids)
      input_masks.append(masks)
      input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]

def nlp_model(bert_base, MAX_SEQ_LEN):
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=bert_base, trainable=True)  
    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")           
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")       
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")

    inputs = [input_ids, input_masks, input_segments] # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs) # BERT outputs
    
    x = Dense(units=768, activation='relu')(pooled_output) # hidden layer 
    x = Dropout(0.15)(x) 
    outputs = Dense(2, activation="softmax")(x) # output layer

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train():
    BATCH_SIZE = 32
    EPOCHS = 3
    MAX_SEQ_LEN = 128

    # model construction (we construct model first inorder to use bert_layer's tokenizer)
    bert_base = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    model = nlp_model(bert_base, MAX_SEQ_LEN) 
    model.summary()
    df = pd.read_csv('train.csv')
    # Create examples for training and testing
    df = df.sample(frac=1) # Shuffle the dataset
    tokenizer = create_tonkenizer(model.layers[3])

    # create training data and testing data
    x_train = convert_sentences_to_features(df['sentence'][:5000], tokenizer, MAX_SEQ_LEN)
    x_valid = convert_sentences_to_features(df['sentence'][6000:7000], tokenizer, MAX_SEQ_LEN)
    x_test = convert_sentences_to_features(df['sentence'][10000:11000], tokenizer, MAX_SEQ_LEN)

    df['polarity'].replace('positive', 1., inplace=True)
    df['polarity'].replace('negative', 0., inplace=True)
    one_hot_encoded = to_categorical(df['polarity'].values)
    y_train = one_hot_encoded[:5000]
    y_valid = one_hot_encoded[6000:7000]
    y_test =  one_hot_encoded[10000:11000]

    optimizer = Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # fit the data to the model
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # save the trained model
    model.save('bert_fine_tuning')

if __name__ == '__main__':
    train()

