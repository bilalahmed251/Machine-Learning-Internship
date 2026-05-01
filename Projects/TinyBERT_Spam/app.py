import streamlit as st
# --- UI CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Cloud AI Detector", page_icon="☁️")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# LOGIC: We are now loading from the CLOUD! 
# This ensures your PC space stays free and your app is portable.
model_path = "b098/tinybert-spam-classifier"

@st.cache_resource # LOGIC: Cache to avoid downloading every time the user clicks a button
def load_model():
    print(f"Loading model from Hugging Face: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# --- UI ---
st.title("☁️ Cloud-Powered Spam Detector")
st.write("This app uses a custom TinyBERT model hosted on the Hugging Face Hub.")

input_sms = st.text_area("Paste message to analyze:", placeholder="Enter text here...")

if st.button('Analyze with AI'):
    if input_sms:
        inputs = tokenizer(input_sms, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probs).item() * 100

        if prediction == 1:
            st.error(f"🚨 **SPAM** ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ **SAFE** ({confidence:.2f}% confidence)")
    else:
        st.warning("Please enter a message.")

st.sidebar.markdown(f"""
### Model Info
- **Source:** [Hugging Face Hub](https://huggingface.co/{model_path})
- **Engine:** TinyBERT (Transformer)
- **Status:** Live & Cloud-Connected
""")
