import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Load the trained model
model = load_model("s2s.keras")

# ✅ Load training data details
datafile = pickle.load(open("training_data.pkl", "rb"))
input_characters = datafile['input_characters']
target_characters = datafile['target_characters']
max_input_length = datafile['max_input_length']
max_target_length = datafile['max_target_length']
num_en_chars = datafile['num_en_chars']
num_dec_chars = datafile['num_dec_chars']

# ✅ Reverse lookup dictionary
reverse_target_char_index = {i: char for i, char in enumerate(target_characters)}

# ✅ Build Encoder Model
enc_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_model = Model(model.input[0], [state_h_enc, state_c_enc])

# ✅ Build Decoder Model
# Create input layers for decoder states (Fixing the error)
decoder_state_input_h = Input(shape=(256,), name="decoder_hidden_state_input")
decoder_state_input_c = Input(shape=(256,), name="decoder_cell_state_input")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Use existing decoder LSTM layer
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    model.input[1], initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]

# Output layer
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)

# Create the new decoder model
decoder_model = Model(
    inputs=[model.input[1]] + decoder_states_inputs,
    outputs=[decoder_outputs] + decoder_states
)

# ✅ Convert Input Text to One-Hot Encoding
def preprocess_text(input_t):
    cv = CountVectorizer(binary=True, analyzer='char')
    en_in_data = []
    pad_en = [1] + [0] * (len(input_characters) - 1)

    cv.fit(input_characters)
    en_in_data.append(cv.transform(list(input_t)).toarray().tolist())

    if len(input_t) < max_input_length:
        for _ in range(max_input_length - len(input_t)):
            en_in_data[0].append(pad_en)

    return np.array(en_in_data, dtype="float32")

# ✅ Function to Translate Text
def translate_text(input_text):
    input_seq = preprocess_text(input_text.lower() + ".")
    states_value = encoder_model.predict(input_seq)

    cv = CountVectorizer(binary=True, analyzer='char')
    cv.fit(target_characters)

    target_seq = np.array([cv.transform(list("\t")).toarray().tolist()], dtype="float32")

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_chars, h, c = decoder_model.predict([target_seq] + states_value)
        char_index = np.argmax(output_chars[0, -1, :])
        text_char = reverse_target_char_index[char_index]
        decoded_sentence += text_char

        if text_char == "\n" or len(decoded_sentence) > max_target_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_dec_chars))
        target_seq[0, 0, char_index] = 1.0
        states_value = [h, c]

    return decoded_sentence.strip()

# ✅ Example Test (Run this script directly)
if __name__ == "__main__":
    while True:
        text = input("Enter English sentence (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        print("French Translation:", translate_text(text))
