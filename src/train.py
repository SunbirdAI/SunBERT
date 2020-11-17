import pandas as pd
from dataset import create_dataloader
from engine import train_fn
from model import SunBERT
from transformers import AdamW
from torch import nn


def run():
    MAX_LEN = 66
    BATCH_SIZE = 32
    EPOCHS = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "Couldn't find GPU device")

    # BERT Specific Pre-processing
    bert_cased = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_cased)

    cls_model = SunBERT(3) # replace hard-coded number of categories
    cls_model = cls_model.to(device)
    # Put the above lines in a config file

    dfx = pd.read_csv("dataset.csv")
    dfx.category = dfx.category.apply(lambda x: 0 if x=="Organic" else 1 if x=="Editorial" else 2) # replace hard-coded categories

    random_seed = 42 
    df_train, df_test = train_test_split(dfx, test_size=0.15, random_state = random_seed)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state = random_seed)

    dataloader_train = create_dataloader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    dataloader_test = create_dataloader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    dataloader_val = create_dataloader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    optimizer = AdamW(cls_model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(dataloader_train) * EPOCHS
    ln_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_func = nn.CrossEntropyLoss().to(device)

    train_fn(cls_model, dataloader_train, loss_func, optimizer, device, ln_scheduler, len(df_train))

if __name__ == '__main__':
    run()
