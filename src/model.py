import json
import transformers
from transformers import BertModel
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])
        classifier = SunBert(len(config["CLASS_NAMES"]))

        classifier.load_state_dict(
                torch.load(config["PRETRAINED_MODEL"], map_location=self.device)
            )

            classifier = classifier.eval()
            self.classifier = classifier.to(self.device)




