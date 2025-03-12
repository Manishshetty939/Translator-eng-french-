import streamlit as st
from inference_model import translate_text

# 🎨 Streamlit UI
st.title("🌍 Language Translator")
st.subheader("Translate English to French using Machine Learning")

# 🚀 User Input
user_input = st.text_area("Enter an English sentence:", "")

# 🛠 Translation Button
if st.button("Translate"):
    if user_input.strip():
        translated_text = translate_text(user_input)
        st.success(f"**French Translation:** {translated_text}")
    else:
        st.warning("⚠️ Please enter a sentence to translate.")
