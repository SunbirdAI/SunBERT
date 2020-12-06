# SunBERT
Sunbird AI BERT style model for text classification.


# Usage

## Installation
Run ``pip install -r requirements.txt``

## Pre-Requisites

This model is built and trained using the [HuggingFace 🤗](https://huggingface.co/) framework. 

Requires a trained Bert Model. Example training notebooks can be found in ``notebooks`` dir
We provide a pretrained model for Covid/Non-Covid Tweet classification

Begin By downloading pretrained model by running:

`` bin/download_model``

## Run Local Inference Server

1. Begin Server by running ``bin/start_server``
1. When server starts on another terminal: Send a request to the endpoint: <br>`` http POST http://localhost:8000/predict text = "Uganda imposes strict lockdown measures" ``
1. Alternatively; You can test the api by running ``bin/test_response``

1. Output should look like:

```
HTTP/1.1 200 OK
content-length: 137
content-type: application/json
date: Sun, 06 Dec 2020 21:24:31 GMT
server: uvicorn

{
    "classification": "Covid",
    "confidence": 0.9988712668418884,
    "probabilities": {
        "Covid": 0.9988712668418884,
        "Non-Covid": 0.0011287190718576312
    }
}

```

