import streamlit as st
from utils.translator import LANGUAGES, load_model, translate

st.set_page_config(page_title="Translator App", page_icon="🌍")

st.title("🌍 Language Translator")

# Select target language
target_lang = st.selectbox(
    "Choose target language:",
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x]
)

# Input text
text_input = st.text_area("Enter text in English:", "")

if st.button("Translate"):
    if text_input.strip():
        with st.spinner("Translating..."):
            model, tokenizer = load_model("en", target_lang)
            output = translate(text_input, model, tokenizer)
        st.success("Translation complete!")
        st.write(f"**{LANGUAGES[target_lang]}:** {output}")
    else:
        st.warning("Please enter some text to translate.")
