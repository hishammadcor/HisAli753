# -*- coding: utf-8 -*-

import pandas as pd

# import data
data = pd.read_csv('sample_data.csv')
print("First 5 rows: \n",data.head(),'\n') # print the first 5 rows
print("Full data summary: \n",data.describe(),'\n') # full describtion and analysis of the data
print("Total number of null values: \n",data.isnull().sum()) # Check for missing values

'''
Data preprocessing phase using SpaCy german model
'''
import spacy
import re

# load spacy german model
nlp = spacy.load('de_core_news_sm')

def preprocess_text(text):
    '''
    This function:
      1. ensure that all the text is only german alphabet.
      2. use spacy lemmatizer fucntion to lemmatize the text tokens, and ensure that it's in lower case.

    Input: Data ['text']

    Return: Lemmatized, lowercase, not-null values and only german alphabet.
    '''

    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', text) # ensure that all the text is german alphabet
    tokens = nlp(text) # load the text to spacy nlp function to be tokenized
    lemmatized_tokens = [token.lemma_.lower() for token in tokens if not token.is_punct] # all the tokens are lemmatized, become in lowercase, and delete all punctuations

    return ' '.join(lemmatized_tokens)


data['text'] = data['text'].apply(preprocess_text)
data = data.dropna() # dropping all the null values in the dataset

'''
Feature extraction using Tf-IDF and model training:

- In this phase I used three models:
    1. Multinominal Naive bayes model >>> not the perfect results, but it's popular in this domain
    2. Random forest classifier >>> higher accuracy and F1 score than Naive
    3. Finetuned the BERT german base model on the data set and use its embiddings to train Random forest and it gives the best results more than 90%

NOTE: I run all the models using the basic parameters.
'''

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib') # save the vectroizer


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Training
NaiveB_model = MultinomialNB() 
NaiveB_model.fit(X_train, y_train)
joblib.dump(NaiveB_model, 'naive_bayes_model.joblib')

# Model Evaluation
y_pred = NaiveB_model.predict(X_test)
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

# Random forest training
RanFor_model = RandomForestClassifier()
RanFor_model.fit(X_train, y_train)
joblib.dump(RanFor_model, 'Random_Forest_model.joblib')

# Model Evaluation
y_pred = RanFor_model.predict(X_test)
print(classification_report(y_test, y_pred))

# BERT model training phase
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Label encoder function: to transform the labels into integers
label_encoder = LabelEncoder()

# Encode labels
data['label'] = label_encoder.fit_transform(data['label'])


texts = data['text'].tolist()
labels = data['label'].tolist() 

train_set, val_set, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# using bert german base model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

# tokenize training and validation sets
train_encodings = tokenizer(train_set, truncation=True, padding=True, max_length=150)
val_encodings = tokenizer(val_set, truncation=True, padding=True, max_length=150)


class GermanDataset(Dataset):
    '''
    This class is used to:
        1. make a dictionary that has the encodded text and its values to be ready for bert
        2. gets the length of the labels
    '''
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = GermanDataset(train_encodings, train_labels)
val_dataset = GermanDataset(val_encodings, val_labels)


model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=len(set(labels)))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()

# Model saving
model.save_pretrained('./fine_tuned_german_bert')

'''
Trainging random forest on the embeddings of the finetuned BERT model
'''
from transformers import BertModel, BertTokenizer

# Load the fine-tuned model and tokenizer
model = BertModel.from_pretrained('./fine_tuned_german_bert')
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

model.eval()

def generate_embeddings(text):
    '''
    This function is resbonsible tokenizing the data[text] into BERT format inputs,
    then, with out the gradient, these inputs goes as an argument to the model, 
    finally, we use the last hidden layer outputs as embeddings.
    '''
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=150)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

embeddings = data['text'].apply(generate_embeddings)

import numpy as np

X = np.vstack(embeddings.values)  # use numpy to stack arrays
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Random_Bert = RandomForestClassifier()
Random_Bert.fit(X_train, y_train)
joblib.dump(Random_Bert, 'Random_Forest_Bert_model.joblib')

# Predictions for RandomForest Bert.
y_pred_bert = Random_Bert.predict(X_test)

print(classification_report(y_test, y_pred_bert))