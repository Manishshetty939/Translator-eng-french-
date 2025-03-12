# Translator-eng-french-

🌍 English-to-French Translator

This project is a Sequence-to-Sequence (Seq2Seq) Language Translator that translates English sentences into French using a deep learning model trained with LSTM (Long Short-Term Memory) networks.

📌 Features

LSTM-based Encoder-Decoder Model trained on an English-French dataset

Streamlit Web Interface for easy user interaction

One-Hot Encoding for text preprocessing

Pre-trained Model for Fast Translations

🚀 Getting Started

1️⃣ Install Dependencies

Ensure you have Python installed, then run:

pip install tensorflow numpy scikit-learn streamlit

2️⃣ Train the Model (Optional)

To train the model from scratch, run:

python langTraining.py

This will create s2s.keras (trained model) and training_data.pkl (character mappings).

3️⃣ Run the Translator

Start the Streamlit web app using:

streamlit run translator.py

🏗 Project Structure

📂 project-root/
├── README.md           # Project documentation
├── langTraining.py     # Model training script
├── inference_model.py  # Model inference (translation) script
├── translator.py       # Streamlit UI for translation
├── s2s.keras           # Pre-trained model (saved after training)
├── training_data.pkl   # Pickle file with character mappings

📖 How It Works

langTraining.py loads and preprocesses English-French sentences, builds the Seq2Seq model, and trains it.

inference_model.py loads the trained model, builds separate encoder and decoder models, and defines the translate_text() function.

translator.py provides a simple web interface using Streamlit, where users can input English sentences and get French translations.
