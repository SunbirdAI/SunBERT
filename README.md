# SunBERT
Sunbird AI BERT style model for text classification.


# Usage

## Installation
Run ``pip install -r requirements.txt``

## Pre-Requisites

Requires a trained Bert Model. Example training notebooks can be found in ``notebooks`` dir
We provide a pretrained model for Covid/Non-Covid Tweet classification

Begin By downloading pretrained model by running:

`` bin/download_model``

## Run Local Inference Server

1. Begin Server by running ``bin/start_server``
1. When server starts on another terminal: Send a request to the endpoint: `` http POST http://localhost:8000/predict text = "Help, I got covid19 !" ``
1. Alternatively; You can test the api by running ``bin/test_response``
1. 