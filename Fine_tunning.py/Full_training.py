# %%
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# %%
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# %%
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# %%
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# %%

# for batch in train_dataloader:
#     break
batch = next(iter(train_dataloader))
{k: v.shape for k, v in batch.items()}

# %%

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# %% 

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# %%
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
# %%

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
# %%

import torch

print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])
# num_of_gpus = torch.cuda.device_count()
# print("The Number of the GPUs are: ", num_of_gpus)

# print("Current GPU", torch.cuda.current_device())

# torch.cuda.device(2)
# torch.cuda.set_device(0)
# print("New Selected GPU", torch.cuda.current_device())

# %%
import os 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# torch.cuda.set_device(0)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
# print(device)
torch.cuda.get_arch_list()
torch.cuda.get_device_properties("cuda:0")
# torch.cuda.get_device_properties()
print("New Selected GPU", torch.cuda.current_device())

device = "cuda:0"
# %%

model.to(device)

# %%
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    print("Epoch: " , epoch)
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
## %%

# %%
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

# %%
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

def training_function():
    accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
# %%
from accelerate import notebook_launcher

notebook_launcher(training_function,num_processes=2)
# %%
