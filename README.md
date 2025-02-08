This code fine-tunes a pre-trained language model to classify expressions from clinical narrative notes regarding alcohol use. The goal is to identify if the expression indicates alcohol use or if it depicts appropriate use or non-use.

Model Overview
The model is built on pre-trained models (e.g., "Bio_ClinicalBERT," "UFNLP/gatortron-base") from Hugging Face's Transformers library, adapted to recognize specific patterns in clinical narratives that relate to alcohol consumption.

Data for Fine-tuning
See: https://huggingface.co/datasets/kartoun/Alcohol_Use_Clinical_Notes_GPT4
This dataset contains 1,500 samples of expressions indicating alcohol use or its negation, generated from clinical narrative notes using OpenAI's ChatGPT 4 model. It's designed to support NLP applications that require the identification of alcohol use references in healthcare records.

Usage
Run the "Fine-tuning" script to handle data loading, model initialization, training, and saving the model outputs. Afterwards, run the "Performance assessment" script to evaluate the model's performance metrics

Output
The trained model and tokenizer are saved in a designated directory, along with performance metrics for review.

Contributing
Feel free to contribute to this project by submitting pull requests or opening issues for any bugs or enhancements you identify.

Acknowledgment
This project utilizes the dataset and pre-trained model developed by Dr. Uri Kartoun.
