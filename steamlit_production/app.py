# Streamlit run app.py in command line or terminal to run the application....

import streamlit as st
import os
import boto3
import torch
from transformers import pipeline


bucket_name = "mlopsmorris"

local_path = "tinybert-sentiment-analysis"
s3_prefix = "ml-models/tinybert-sentiment-analysis/"

s3 = boto3.client("s3")
def download_dir(local_path, s3_prefix):
                 os.makedirs(local_path, exist_ok=True)
                 paginator = s3.get_paginator("list_objects_v2")
                 for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
                    if "Contents" in result:
                        for key in result["Contents"]:
                            s3_key = key["Key"]

                            local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                            # os.makedirs(os.path.dirname(local_file), exist_ok=True)

                            s3.download_file(bucket_name, s3_key, local_file)


st.title("Machine Learning Model Deployment…")

# Download button
if st.button("Download Model"):
    with st.spinner("Downloading model…"):
        download_dir(local_path, s3_prefix)
        st.success("Downloaded!")

# Text input & prediction
text = st.text_area("Enter your review", "Type here…")

# Initialize classifier just once
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline(
    "text-classification",
    model="tinybert-sentiment-analysis",
    device=0 if device=="cuda" else -1
)

# Only call and write when they hit Predict
if st.button("Predict"):
    with st.spinner("Predicting…"):
        result = classifier(text)           # local var
    st.write(result)            

