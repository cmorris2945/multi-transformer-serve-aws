import os
import warnings
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd

# 1. Ensure HF token is visible
# If you exported HUGGINGFACEHUB_API_TOKEN, transformers will pick it up.
# Otherwise, you can also set:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load data
data = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv")
dataset = Dataset.from_pandas(data).train_test_split(test_size=0.3)
label2id = {"negative": 0, "positive": 1}
id2label = {v: k for k, v in label2id.items()}
dataset = dataset.map(lambda x: {"label": label2id[x["sentiment"]]})

# 3. Instantiate tokenizer **with** your token
model_ckpt = 'huawei-noah/TinyBERT_General_4L_312D' 
tokenizer = AutoTokenizer.from_pretrained(
    model_ckpt,
    use_fast=True,
    use_auth_token=True  # attaches HUGGINGFACEHUB_API_TOKEN
)

def tokenize(batch):
    return tokenizer(batch["review"], padding=True, truncation=True, max_length=300)

dataset = dataset.map(tokenize, batched=True)

# 4. Instantiate model **with** your token
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    use_auth_token=True
).to(device)

# 5. Trainer setup
args = TrainingArguments(
    output_dir="train_dir",
    overwrite_output_dir=True,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)


trainer.train()


import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_materics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predications= predictions, references= labels)
    return accuracy.compute(predictions=predictions, references= labels) 
trainer.evaluate()



## Model Save and Load for Inference....

trainer.save_model("tinybert-sentiment-analysis")

data = ['this movie was horrible, the plot was really boring. acting was okay,',
        'the movie is really sucked. there is not plot and acting was bad',
        'what a beautiful movie. great plot. acting was good. will see it again.']

from transformers import pipeline
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = pipeline("text-classification", model= "tinybert-sentiment-analysis", device=device)

classifier(data)


## PUSH MODEL TO AWS S3 BUCKET STORAGE....


import boto3

s3 = boto3.client("s3")

bucket_name = "mlopsmorris"

def create_bucket(bucket_name):
    response = s3.list_buckets()
    buckets = [buck['Name'] for buck in response['Buckets']]

    if bucket_name not in buckets:

        s3.create_bucket(Bucket=bucket_name)
        print("Bucket has been created")

    else:
        print("Bucket is already creates 'Jackass!' Go use that one....")


create_bucket(bucket_name)

#### Upload the model I built and trained to AWS S3 bucket,
##  and call it....ml-models/tinyBERT-sentiment-analysis....


import os
import boto3
s3 = boto3.client('s3')
bucket_name = 'mlopsmorris'
def upload_directory(directory_path, s3_prefix):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file).replace('\\', '/')
            relpath = os.path.relpath(file_path, directory_path)
           
            s3_key = os.path.join(s3_prefix, relpath).replace('\\','/')
            s3.upload_file(file_path, bucket_name, s3_key)

            

upload_directory("aws_sentiment_program", "ml-models/tinyBERT-sentiment-analysis")

