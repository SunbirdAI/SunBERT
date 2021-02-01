import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertTokenizer



class create_dataset(Dataset):
    """Class to create dataset 

    Args:
        Dataset (Dataset): Sunbert dataset
    """
    def __init__(self, text, targets, tokenizer, max_len):
        self.text = text
        self.targets = targets 
        self.tokenizer = tokenizer 
        self.max_len = max_len 

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        """Gets a single instance of dataset

        Args:
            item (item): usually the id

        Returns:
            dict: Dictionary of: input_text, targets, input ids, and attention mask
        """
        text = str(self.text[item])
        target = self.targets[item]

        encoder = self.tokenizer.encode_plus(
            text,
            truncation = True,
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = 'pt'
                )
        return {
            'input_text': text,
            'targets':torch.tensor(target, dtype = torch.long),
            'input_ids': encoder['input_ids'].flatten(),
            'attention_mask': encoder['attention_mask'].flatten()

                }


def create_dataloader(df, tokenizer, max_len, batch_size):
    """Function to create data loader

    Args:
        df (Dataframe): Pandas Dataframe
        tokenizer (tokenizer): Huggingface bert tokenizer
        max_len (int): Maximum Length of tokens 
        batch_size (int): Batch size 

    Returns:
        DataLoader: Data Loader used as inout to train the model
    """
    ds = create_dataset(
            text = df.text.to_numpy(),
            targets = df.category.to_numpy(),
            tokenizer = tokenizer,
            max_len = max_len
            )

    return DataLoader(ds, batch_size, num_workers=4)





