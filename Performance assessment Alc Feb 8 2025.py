import os
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

# Configuration and initialization
WORKING_DIR = 'C:/Temp/Alc/'
DATA_FILE = 'Holdout_Set_Alcohol_Use_Clinical_Notes.csv' # Get the file from: https://huggingface.co/datasets/kartoun/Alcohol_Use_Clinical_Notes_GPT4
MODEL_PATH = 'kartoun/Bio_ClinicalBERT_for_Alcohol_Use_Classification' # Fine-tuned model at Hugging Face.
#MODEL_PATH = 'kartoun/gatortron-base_for_Alcohol_Use_Classification'
CACHE_DIR = "D:/HuggingFaceCache" # Store the pre-trained model on an external hard-drive.
login(token='hf_tbd') # Get your own token from Hugging Face.

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)

# Define the prediction function using the loaded model and tokenizer
def predict_using_loaded_model(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction_idx = torch.argmax(probs, dim=-1).item()
    predicted_label = model.config.id2label[prediction_idx]
    return predicted_label

# Set label configuration for clarity
model.config.id2label = {0: 0, 1: 1}
model.config.label2id = {0: 0, 1: 1}

# Predict a sample text
text = "Patient consumes alcohol again."
prediction = predict_using_loaded_model(text, model, tokenizer)
print("Predicted Label:", prediction)

# Predict a sample text
text = "Patient denies any use of alcohol."
prediction = predict_using_loaded_model(text, model, tokenizer)
print("Predicted Label:", prediction)

# Load and prepare data
data_path = os.path.join(WORKING_DIR, DATA_FILE)
df = pd.read_csv(data_path)
test_sentences = dict(zip(df['Blob'], df['Label']))

# Display first 5 entries for sanity check
for sentence, label in list(test_sentences.items())[:5]:
    print(f"'{sentence}': '{label}'")

# Initialize lists for true labels and predictions
true_labels, predictions = [], []

# Predict over the dataset and collect results
for sentence, true_label in test_sentences.items():
    predicted_label = predict_using_loaded_model(sentence, model, tokenizer)
    true_labels.append(true_label)
    predictions.append(predicted_label)
    print(f"Sentence: {sentence}\nPredicted: {predicted_label}, True: {true_label}\n")

# Confusion matrix calculation and visualization
conf_matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.config.id2label.values(), yticklabels=model.config.id2label.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC-AUC calculation
fpr, tpr, thresholds = roc_curve(true_labels, predictions)
roc_auc_value = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

auc_value = roc_auc_score(true_labels, predictions, average=None, multi_class='ovo')
fpr, tpr, thresholds = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)
with open(WORKING_DIR + 'results/roc_values.txt', 'w') as file:
    file.write('False Positive Rate, True Positive Rate, Thresholds\n')
    for f, t, thr in zip(fpr, tpr, thresholds):
        file.write(f"{f}, {t}, {thr}\n")

# Detailed metrics calculation
accuracy = accuracy_score(true_labels, predictions)
precision, recall, fscore, support = precision_recall_fscore_support(true_labels, predictions, average='weighted')

# Save metrics to file
metrics_file_path = os.path.join(WORKING_DIR, 'results', 'evaluation_metrics.txt')
with open(metrics_file_path, 'w') as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F-score: {fscore:.4f}\n")
    file.write(f"ROC AUC: {roc_auc_value:.4f}\n")

# Detailed metrics calculation for each class
precision, recall, fscore, support = precision_recall_fscore_support(true_labels, predictions)

# Opening the file with append permission
with open(metrics_file_path, 'a') as file:    
    file.write("Class-wise metrics:\n")
    for i, (prec, rec, fs, supp) in enumerate(zip(precision, recall, fscore, support)):
        file.write(f"Class {i} - Precision: {prec:.4f}, Recall: {rec:.4f}, F-score: {fs:.4f}, Support: {supp}\n")  