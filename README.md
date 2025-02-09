**Acknowledgment:** This project utilizes the dataset and fine-tuned models developed by Dr. Uri Kartoun (https://urikartoun.com/).

**Goal:** This code fine-tunes a pre-trained language model to classify expressions from clinical narrative notes regarding alcohol use. The goal is to identify if the expression indicates alcohol use or if it depicts appropriate use or non-use.

**Model Overview:** The model is built on pre-trained models (e.g., "emilyalsentzer/Bio_ClinicalBERT," "UFNLP/gatortron-base") from Hugging Face's Transformers library, adapted to recognize specific patterns in clinical narratives that relate to alcohol consumption.

**Access:** https://huggingface.co/kartoun/Bio_ClinicalBERT_for_Alcohol_Use_Classification (108M parameters), https://huggingface.co/kartoun/gatortron-base_for_Alcohol_Use_Classification (355M parameters).

**Classification performance using a held-out set:**

![test text](https://github.com/kartoun/alcohol_use_classification_llms/blob/main/images/ROC%20Feb%209%202025.png)

**Data for Fine-tuning** https://huggingface.co/datasets/kartoun/Alcohol_Use_Clinical_Notes_GPT4
This dataset contains 1,500 samples of expressions indicating alcohol use or its negation, generated from clinical narrative notes using OpenAI's ChatGPT 4 model. It's designed to support NLP applications that require the identification of alcohol use references in healthcare records.

**Usage:** Run the "Fine-tuning" script to handle data loading, model initialization, training, and saving the model outputs. Afterwards, run the "Performance assessment" script to evaluate the model's performance metrics. You can query the model with a single text blob and get a classification value (0 or 1).

**Output:** The fine-tuned model and tokenizer are saved in a designated directory, along with performance metrics for review.

**Generalizability:** The source code can be adapted to fine-tune pre-trained clinical models for a wide range of classification tasks beyond just alcohol use classification. This includes tasks like identifying Social Determinants of Health, as well as multi-label classifications such as smoking status (current, past, never) or more complex scenarios like classifying one of the seven classes of pancreatic cancer, all with minimal adjustments to the code and training set.

**Contributing:** Feel free to contribute to this project by submitting pull requests or opening issues for any bugs or enhancements you identify.
