import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# --- LOGIC: CHOICE OF MODEL ---
# Why TinyBERT? In industry, we don't always use the biggest model (like GPT). 
# TinyBERT is optimized for speed. Using this shows recruiters you understand 
# "Production Constraints" (running models on phones or cheap servers).
model_name = "huawei-noah/TinyBERT_General_4L_312D"

def main():
    # --- LOGIC: DATA LOADING ---
    # We load 'spam.csv'. We use latin-1 because SMS data often contains 
    # special characters or emojis that standard UTF-8 might fail to read.
    csv_path = 'spam.csv'
    if not os.path.exists(csv_path):
        print("Error: spam.csv not found in the current directory.")
        return

    df = pd.read_csv(csv_path, encoding='latin-1')
    
    # If the CSV has 'label' and 'text', use them. Otherwise assume v1, v2.
    if 'label' in df.columns and 'text' in df.columns:
        df = df[['label', 'text']]
    else:
        df = df[['v1', 'v2']] 
        df.columns = ['label', 'text']

    # --- LOGIC: LABEL ENCODING ---
    # Deep Learning models strictly work with numbers. We convert 'ham' -> 0 and 'spam' -> 1.
    # This is called Binary Classification.
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label']) 

    # --- LOGIC: TOKENIZATION ---
    # Traditional ML (like your old Naive Bayes) used TF-IDF (word counting).
    # TinyBERT uses "Sub-word Tokenization". It breaks words like "running" into "run" + "##ning".
    # This helps the model understand the root meaning of words it hasn't seen before.
    print(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- LOGIC: PADDING & TRUNCATION ---
    # Models process data in "batches" (like a tray of cookies). Every "cookie" (message) 
    # must be the exact same size for the math to work. 
    # 'padding' adds zeros to short messages, 'truncation' cuts long ones.
    def tokenize_func(texts):
        return tokenizer(texts, padding='max_length', truncation=True, max_length=128)

    # --- LOGIC: DATA SPLITTING ---
    # 20% test size is standard. It ensures we have enough data to prove the model works 
    # on messages it has NEVER seen during training.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    print("Tokenizing data...")
    train_encodings = tokenize_func(train_texts)
    val_encodings = tokenize_func(val_texts)

    # --- LOGIC: PYTORCH DATASET ---
    # This class is like a "Delivery Truck". It packages the text-numbers (encodings) 
    # and the target-labels together so the Model can consume them one by one.
    class SpamDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SpamDataset(train_encodings, train_labels)
    val_dataset = SpamDataset(val_encodings, val_labels)

    # --- LOGIC: THE BRAIN (MODEL) ---
    # num_labels=2 is critical. It sets up the last layer of the neural network 
    # to output two probabilities: one for Ham, one for Spam.
    print(f"Loading Model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # --- LOGIC: TRAINING SETTINGS ---
    # 'num_train_epochs=3': We show the data to the model 3 times. Too many times leads 
    # to "Overfitting" (memorizing instead of learning).
    # 'fp16=False': We use standard precision for better compatibility on all CPUs.
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch", # Calculate accuracy after every full pass.
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # --- LOGIC: THE TRAINER ---
    # The Trainer is the "Coach". It manages the training loop, calculates the loss 
    # (how wrong the model is), and updates the model's weights to make it "smarter".
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("--- Starting Training ---")
    # This will take a few minutes. It's the "Learning" phase.
    trainer.train()

    # --- LOGIC: SAVING THE FRUITS ---
    # After training, we save the "Smart" version of the model so we can 
    # use it in your Streamlit app without retraining.
    print("Saving the fine-tuned TinyBERT model...")
    model.save_pretrained("./fine_tuned_tinybert")
    tokenizer.save_pretrained("./fine_tuned_tinybert")
    print("Done! You can now use './fine_tuned_tinybert' in your Streamlit app.")

if __name__ == "__main__":
    main()
