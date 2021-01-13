# SunBERT
Sunbird AI BERT style model for text classification.


# Usage

## Install requirements

Run ``pip install -r requirements.txt``

## Pre-Requisites

This model is built and trained using the [HuggingFace 🤗](https://huggingface.co/) framework. 

It requires a trained Bert Model. Example training notebooks can be found in the ``notebooks`` folder.
We provide a pretrained model for Covid/Non-Covid Tweet classification.

Begin by downloading pretrained model:

* First, create a folder called ``assets`` in this project's root directory:<br>``mkdir assets``

* Then download the model by running ``bin/download_model``


## Run Local Inference Server

1. Begin the server by running ``bin/start_server``
1. When the server starts, open another terminal and send a request to the endpoint as shown here: <br>``http POST http://localhost:8000/predict text="Uganda imposes strict lockdown measures" ``

1. Alternatively, you can test the api by running ``bin/test_request``

1. The output should look like this:

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

## API Documentation
Check the docs here: `http://localhost:8000/docs` to see the full schema. These docs can also be used as a quick way to test `POST` requests.

## Troubleshooting
If you run into errors when executing the commands ``bin/download_model``, ``bin/start_server``, or ``bin/test_request``, try setting the write permissions of the file as shown in the example below:<br>``chmod +x bin/download_model``
