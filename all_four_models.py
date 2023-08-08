##RUN THIS FILE TO PERFORM ALL MODEL TESTING SEQUENTIALLY
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
SUBSET_PERCENTAGE = 0.06 #SNLI TRAIN is 570k large 1=100%, 0.5=50%, ...

PATH_TO_GLOVE = "/home/monty/RyersonGraduateStudies/CP8210/Final_Project/natural_language_inference/glove.6B.100d.txt"

PATH_TO_OUTPUT_FILE = "output.txt"

from datasets import load_dataset, Dataset, DatasetDict

from tqdm import tqdm
import statistics
import numpy as np
from datasets import concatenate_datasets
import spacy
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


from nltk.tokenize import word_tokenize
import numpy as np
from keras.utils import to_categorical
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-mpnet-base-v2', device="cuda")


from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, CSVLogger
import tensorflow

from sklearn.metrics import classification_report


#Actually import dataset into RAM
print("Loading dataset")
dataset_train = load_dataset("snli", split="train")[:int(SUBSET_PERCENTAGE*570000)]
print("Loaded Train")
dataset_test = load_dataset("snli", split="test")[:1000]
print("Loaded Train")
# dataset_valid = load_dataset("snli", split="validation")[:]


dataset = DatasetDict({"train":Dataset.from_dict(dataset_train), "test":Dataset.from_dict(dataset_test)})#, "validation":Dataset.from_dict(dataset_valid)})
print("Done Loading dataset")

#VOCAB SIZE

print("Finding vocab size")
full_corpus = concatenate_datasets([dataset["train"], dataset["test"]])#, dataset["validation"]])
text_full_corpus = full_corpus["premise"]+full_corpus["hypothesis"]
text_full_corpus_string = " ".join(text_full_corpus).lower()
# count the number of unique tokens
vocab_size = len(set(text_full_corpus_string.split()))
print("Vocabulary size:", vocab_size)

print("Finding max length")
max_len = 0
for text in tqdm(text_full_corpus):
    current_length = len(text.split())
    if current_length > max_len:
        max_len = current_length
print("Max length", max_len)

print("Cleaning data")
print("Filtering -1 labels from dataset")
filtered_dataset = dataset.filter(lambda example: example['label'] != -1)# and example['label'] != 1)
print("Setting new vocab size")
full_corpus = concatenate_datasets([filtered_dataset["train"], filtered_dataset["test"]])#, filtered_dataset["validation"]])
text_full_corpus = full_corpus["premise"]+full_corpus["hypothesis"]
text_full_corpus = " ".join(text_full_corpus).lower()
# count the number of unique tokens
vocab_size = len(set(text_full_corpus.split(" ")))
print("Vocabulary size:", vocab_size)

print("Setting new max length")
for text in tqdm(text_full_corpus):
    current_length = len(text.split())
    if current_length > max_len:
        max_len = current_length
print("Max length", max_len)


print("Preprocessing data")

print("Embedding sentences for basic model")
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
sbert_model.max_seq_length = 100
train_prem_sentEmb = sbert_model.encode(filtered_dataset["train"]["premise"], show_progress_bar=True)
train_hyp_sentEmb = sbert_model.encode(filtered_dataset["train"]["hypothesis"], show_progress_bar=True)

test_prem_sentEmb = sbert_model.encode(filtered_dataset["test"]["premise"], show_progress_bar=True)
test_hyp_sentEmb = sbert_model.encode(filtered_dataset["test"]["hypothesis"], show_progress_bar=True)


print("Tokenizing and padding data")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(full_corpus["premise"]+full_corpus["hypothesis"])
train_premise_sequences = tokenizer.texts_to_sequences(filtered_dataset["train"]["premise"])
train_hypothesis_sequences = tokenizer.texts_to_sequences(filtered_dataset["train"]["hypothesis"])
test_premise_sequences = tokenizer.texts_to_sequences(filtered_dataset["test"]["premise"])
test_hypothesis_sequences = tokenizer.texts_to_sequences(filtered_dataset["test"]["hypothesis"])

padded_train_premise_sequences = pad_sequences(train_premise_sequences, padding='post', maxlen=max_len)
padded_train_hypothesis_sequences = pad_sequences(train_hypothesis_sequences, padding='post', maxlen=max_len)

padded_test_premise_sequences = pad_sequences(test_premise_sequences, padding='post', maxlen=max_len)
padded_test_hypothesis_sequences = pad_sequences(test_hypothesis_sequences, padding='post', maxlen=max_len)
tokenizer.word_index["NULL"] = 0
tokenizer.index_word[0] = "NULL"

print("Done")

print("Embedding text woth GloVe")
print("Loading GloVe pretrained model")
#FROM : https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
embeddings_dict = {}

with open(PATH_TO_GLOVE, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
print("Done")



print("Mapping embeddings to tokens")
#Let's map the embeddings to our own tokenized words!
# Create a dictionary that maps tokens to their embeddings
embedding_dict_tokenized = {}
for token in tqdm(tokenizer.word_index):
    if token in list(embeddings_dict.keys()):
        embedding_dict_tokenized[token] = embeddings_dict[token]
    else:
        embedding_dict_tokenized[token] = np.random.rand(100)
embedding_dict_tokenized["NULL"] = np.zeros(100)
print("Done")

print("Embedding training sequences")
def embedded_sequence(sequences, embedding_dict_tokenized):
    embedded_sequences = []
    for seq in tqdm(sequences):
        embedded_sequence = []
        for token in seq:
            embedded_sequence.append(embedding_dict_tokenized[tokenizer.index_word[token]])
        embedded_sequences.append(embedded_sequence)
    return embedded_sequences
train_hyp = embedded_sequence(padded_train_hypothesis_sequences, embedding_dict_tokenized)
train_prem = embedded_sequence(padded_train_premise_sequences, embedding_dict_tokenized)
print("Done")
print("Embedding testing sequences")
test_hyp = embedded_sequence(padded_test_hypothesis_sequences, embedding_dict_tokenized)
test_prem = embedded_sequence(padded_test_premise_sequences, embedding_dict_tokenized)
print("Done")
print("One hot encoding labels")
y_train = to_categorical(filtered_dataset["train"]["label"], num_classes=3)
y_test = to_categorical(filtered_dataset["test"]["label"], num_classes=3)
print("Done")

print("Creating model")

def build_conditional_LSTM_SNLI_model(): #This model uses gloVe embeddings
    input_premise = Input(shape=(max_len, 100))
    input_hypothesis = Input(shape=(max_len, 100))
    lstm_premise = LSTM(units=64, return_state=True)(input_premise)
    lstm_hypothesis = LSTM(units=64)(input_hypothesis, initial_state=lstm_premise[1:])
    output = Dense(units=3, activation="softmax")(lstm_hypothesis)
    model = tensorflow.keras.models.Model(inputs=[input_premise, input_hypothesis], outputs=output)
    return model

def build_bowman_SNLI_model(): #This model uses all-mpnet-base-v2 sentence embeddings
    input_premise = Input(shape=(768))
    input_hypothesis = Input(shape=(768))
    concat_premise_hypothesis = Concatenate()([input_premise, input_hypothesis])
    tanh1 = Dense(units=200, activation="tanh", )(concat_premise_hypothesis)
    tanh2 = Dense(units=200, activation="tanh")(tanh1)
    tanh3 = Dense(units=200, activation="tanh")(tanh2)
    output = Dense(units=3, activation="softmax")(tanh3)
    model = tensorflow.keras.models.Model(inputs=[input_premise, input_hypothesis], outputs=output)
    return model

def build_bowman_LSTM_model(): #This model uses LSTM sentence embeddings
    input_premise = Input(shape=(max_len, 100))
    input_hypothesis = Input(shape=(max_len, 100))
    premise_lstm = LSTM(units=100)(input_premise)
    hypothesis_lstm = LSTM(units=100)(input_hypothesis)
    concat_premise_hypothesis = Concatenate()([premise_lstm, hypothesis_lstm])
    tanh1 = Dense(units=200, activation="tanh", )(concat_premise_hypothesis)
    tanh2 = Dense(units=200, activation="tanh")(tanh1)
    tanh3 = Dense(units=200, activation="tanh")(tanh2)
    output = Dense(units=3, activation="softmax")(tanh3)
    model = tensorflow.keras.models.Model(inputs=[input_premise, input_hypothesis], outputs=output)
    return model

def build_wordword_attention_SNLI_model(): #This is still a TODO!
    input_premise = Input(shape=(max_len, 100))
    input_hypothesis = Input(shape=(max_len, 100))
    lstm_premise = LSTM(units=64, return_sequences=True)(input_premise) #key
    lstm_hypothesis = LSTM(units=64)(input_hypothesis) #query
    attn = tensorflow.keras.layers.Attention()([lstm_premise, lstm_hypothesis])
    attn_flat = tensorflow.keras.layers.Flatten()(attn)
    output = Dense(units=3, activation="softmax")(attn_flat)
    model = tensorflow.keras.models.Model(inputs=[input_premise, input_hypothesis], outputs=output)
    return model

def train_test_performancemetrics(model, trainX, trainY, testX, testY, optimizer, model_name:str, batch=None): #Trains, tests, and measures performance for a model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Training")

    early_stop = EarlyStopping(monitor='loss', patience=5, min_delta=0.0005, restore_best_weights=True)
    csv_logger = CSVLogger('training.log')
    model.fit(trainX, trainY, epochs=10, callbacks=[early_stop, csv_logger], batch_size=batch)
    print("Done")
    print("Testing")
    pred = model.predict(testX)
    correct_labels = [np.argmax(x) for x in testY]
    pred_labels = [np.argmax(x) for x in pred]
    print("Done")
    print("Performance")
    report = classification_report(correct_labels, pred_labels)
    with open(PATH_TO_OUTPUT_FILE, mode='a+') as f:
        f.write(f"{model_name}\n")
        f.write(report)


print("Performing basic model with MPnet sentence embeddings")
train_test_performancemetrics(build_bowman_SNLI_model(), [np.array(train_prem_sentEmb), (np.array(train_hyp_sentEmb))], 
                              y_train, [np.array(test_prem_sentEmb), 
                                        (np.array(test_hyp_sentEmb))],
                                          y_test, tensorflow.keras.optimizers.SGD(), 
                                "Basic Bowman model with MPnet Sentence Embeddings")

print("Performing basic model with LSTM based sentence embeddings")
train_test_performancemetrics(build_bowman_LSTM_model(),  [np.array(train_prem),
                                                            (np.array(train_hyp))], 
                              y_train, [np.array(test_prem), (np.array(test_hyp))],
                                y_test, tensorflow.keras.optimizers.SGD(),
                                "Basic Bowman Model with LSTM sentence embeddings",
                                  batch = 16)

print("Performing conditional LSTM model")
train_test_performancemetrics(build_conditional_LSTM_SNLI_model(),  [np.array(train_prem), (np.array(train_hyp))], 
                              y_train,[np.array(test_prem), (np.array(test_hyp))], y_test, tensorflow.keras.optimizers.Adam(), "Conditional LSTM model")

print("Performing attention based model")
train_test_performancemetrics(build_wordword_attention_SNLI_model(),  [np.array(train_prem), (np.array(train_hyp))], 
                              y_train, [np.array(test_prem), (np.array(test_hyp))], y_test, tensorflow.keras.optimizers.Adam(), "Attention-based model")
