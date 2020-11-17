import transformers 
from transformers import BertModel
from torch import nn


class SunBERT(nn.Module):
    def __init__(self, n_classes):
        super(SunBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_cased)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
                )

        output = self.drop(pooled_output)

        return self.out(output)
