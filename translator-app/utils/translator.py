from transformers import MarianMTModel, MarianTokenizer

# Dictionary of available languages
LANGUAGES = {
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese'
}

def load_model(src_lang="en", tgt_lang="fr"):
    """Load MarianMT model for a given source and target language."""
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate(text, model, tokenizer):
    """Translate text using the loaded model and tokenizer."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
