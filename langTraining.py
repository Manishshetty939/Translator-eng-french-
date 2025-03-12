import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Dataset variables
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# ✅ Load dataset (file_path is already resolved)
with open('data/eng-french.txt', 'r', encoding='utf-8') as f:
    rows = f.read().strip().split('\n')

# ✅ Process first 10,000 rows safely
for row in rows[:10000]:
    parts = row.split('\t')
    if len(parts) < 2:
        continue  # Skip empty or malformed rows
    input_text, target_text = parts[0], parts[1]

    target_text = '\t' + target_text + '\n'  # Add start/end markers
    input_texts.append(input_text.lower())
    target_texts.append(target_text.lower())

    input_characters.update(list(input_text.lower()))
    target_characters.update(list(target_text.lower()))


input_characters = sorted(input_characters)
target_characters = sorted(target_characters)


num_en_chars = len(input_characters)
num_dec_chars = len(target_characters)
max_input_length = max(len(i) for i in input_texts)
max_target_length = max(len(i) for i in target_texts)


def one_hot_encoding(input_texts, target_texts):
    cv_inp = CountVectorizer(binary=True, analyzer='char')
    cv_tar = CountVectorizer(binary=True, analyzer='char')

   
    cv_inp.fit(input_characters)
    cv_tar.fit(target_characters)

    en_in_data = []
    dec_in_data = []
    dec_tr_data = []

    for input_t, target_t in zip(input_texts, target_texts):
        en_vec = cv_inp.transform(list(input_t)).toarray()
        dec_vec = cv_tar.transform(list(target_t)).toarray()
        dec_tr_vec = cv_tar.transform(list(target_t)[1:]).toarray()

        
        en_vec = np.pad(en_vec, ((0, max_input_length - len(en_vec)), (0, 0)), mode='constant')
        dec_vec = np.pad(dec_vec, ((0, max_target_length - len(dec_vec)), (0, 0)), mode='constant')
        dec_tr_vec = np.pad(dec_tr_vec, ((0, max_target_length - len(dec_tr_vec)), (0, 0)), mode='constant')

        en_in_data.append(en_vec)
        dec_in_data.append(dec_vec)
        dec_tr_data.append(dec_tr_vec)

    return np.array(en_in_data, dtype="float32"), np.array(dec_in_data, dtype="float32"), np.array(dec_tr_data, dtype="float32")


en_in_data, dec_in_data, dec_tr_data = one_hot_encoding(input_texts, target_texts)


en_inputs = Input(shape=(None, num_en_chars))
encoder = LSTM(256, return_state=True)
en_outputs, state_h, state_c = encoder(en_inputs)
en_states = [state_h, state_c]


dec_inputs = Input(shape=(None, num_dec_chars))
dec_lstm = LSTM(256, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(dec_inputs, initial_state=en_states)
dec_dense = Dense(num_dec_chars, activation="softmax")
dec_outputs = dec_dense(dec_outputs)


model = Model([en_inputs, dec_inputs], dec_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(
    [en_in_data, dec_in_data], dec_tr_data, 
    batch_size=64, epochs=50, validation_split=0.2
)


model.save("s2s.keras")
pickle.dump({
    'input_characters': input_characters,
    'target_characters': target_characters,
    'max_input_length': max_input_length,
    'max_target_length': max_target_length,
    'num_en_chars': num_en_chars,
    'num_dec_chars': num_dec_chars
}, open("training_data.pkl", "wb"))
