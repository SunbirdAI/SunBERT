from transformers import AdamW
from torch import nn
from dataset import create_dataloader
from model import  SunBERT
import pandas as pd


def train_fn(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = cls_model.train()

    losses = []
    correct_preds = 0

    for x in data_loader:
        input_ids =  x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        targets =  x["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_preds += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_preds.double()/n_examples, np.mean(losses)

if __name__ == '__main__':
    BATCH_SIZE = 32
    EPOCHS = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "Couldn't find GPU device")

    dfx = pd.read_csv("dataset.csv")
    dfx.category = dfx.category.apply(lambda x: 0 if x=="Organic" else 1 if x=="Editorial" else 2)

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

    train_fn(SunBERT, dataloader_train, loss_func, optimizer, device, ln_scheduler, len(df_train))

