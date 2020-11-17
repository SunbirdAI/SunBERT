from torch import nn


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
    
def evaluate_model(model, data_loader, loss_fn, device, n_examples):
  model = cls_model.eval()
  
  losses = []
  correct_preds = 0

  with torch.no_grad():

    for x in data_loader:
      input_ids = x["input_ids"].to(device)
      attention_mask = x["attention_mask"].to(device)
      targets = x["targets"].to(device)

      outputs =  model(input_ids = input_ids, attention_mask = attention_mask)
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_preds += torch.sum(preds == targets)
      losses.append(loss.item())

    return correct_preds.double()/n_examples, np.mean(losses)
