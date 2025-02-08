import os
import datetime
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Define directories and paths
CACHE_DIR = "D:/HuggingFaceCache" # Store the pre-trained model on an external hard-drive.
WORKING_DIR = 'C:/Temp/Alc/'
DATA_FILE = 'Training_Set_Alcohol_Use_Clinical_Notes.csv' # Get the file from: https://huggingface.co/datasets/kartoun/Alcohol_Use_Clinical_Notes_GPT4
MODEL_PATH = 'emilyalsentzer/Bio_ClinicalBERT' #UFNLP/gatortron-base

# Set seed for reproducibility
np.random.seed(51)

# Load the dataset
data_path = os.path.join(WORKING_DIR, DATA_FILE)
df = pd.read_csv(data_path, encoding='ISO-8859-1')
print(df.head())
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Convert the DataFrame into a Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, use_auth_token=True)

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples['Blob'], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to the dataset
dataset = dataset.map(tokenize_function, batched=True)

# Define label mapping and conversion function
label_mapping = {0: 0, 1: 1}

def label_to_id(example):
    example['labels'] = label_mapping[example['Label']]
    return example

# Apply label conversion to the dataset
dataset = dataset.map(label_to_id)

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2, cache_dir=CACHE_DIR)

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(WORKING_DIR, 'results'),
    num_train_epochs=5,
    per_device_train_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir=os.path.join(WORKING_DIR, 'logs'),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    load_best_model_at_end=True,
    learning_rate=2e-5
)

# Split the dataset into training and evaluation sets
train_indices, test_indices = train_test_split(
    np.arange(len(dataset['labels'])),
    test_size=0.1,
    stratify=dataset['labels']
)

train_dataset = dataset.select(train_indices)
eval_dataset = dataset.select(test_indices)

# Define the metric computation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(labels, p.predictions[:, 1])
        metrics['roc_auc'] = roc_auc
    
    return metrics

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
now = datetime.datetime.now()
formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
output_dir = os.path.join(WORKING_DIR, 'results', 'model_' + formatted_date_time)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training metrics
train_metrics = trainer.state.log_history
metrics_df = pd.DataFrame(train_metrics)
metrics_file = os.path.join(WORKING_DIR, 'results', 'training_metrics_' + formatted_date_time + '.csv')
metrics_df.to_csv(metrics_file, index=False)
