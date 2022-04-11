# -*- encoding:utf-8 -*-
#https://blog.csdn.net/pearl8899/article/details/116353751
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# tokenizer用来对文本进行编码
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 训练数据
train = {
    'text': [' 测试good',
             '美团 学习',
             ' 测试good',
             '美团 学习',
             ' 测试good',
             '美团 学习',
             ' 测试good',
             '美团 学习'],
    'target': [0, 1, 0, 1, 0, 1, 0, 1],
}

# Get text values and labels
text_values = train['text']
labels = train['target']

print('Original Text : ', text_values[0])
print('Tokenized Ids: ', tokenizer.encode(text_values[0], add_special_tokens=True))
print('Tokenized Text: ', tokenizer.decode(tokenizer.encode(text_values[0], add_special_tokens=True)))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[0])))


# Function to get token ids for a list of texts
def encode_fn(text_list):
    all_input_ids = []
    for text in text_list:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=160,  # 设定最大文本长度
            pad_to_max_length=True,  # pad到最大的长度
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


# 对训练数据进行编码
all_input_ids = encode_fn(text_values)
labels = torch.tensor(labels)

# 训练参数定义
epochs = 1
batch_size = 1

# Split data into train and validation
dataset = TensorDataset(all_input_ids, labels)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained BERT model， num_labels=2表示类别是2
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2, output_attentions=False,
                                                      output_hidden_states=True)
#print(model)
# model.cuda()

# create optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
# 表示学习率预热num_warmup_steps步后，再按照指定的学习率去更新参数
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

flag = False
total_batch, last_improve = 0, 0
require_improvement = 1000
dev_best_loss = float('inf')#跟踪验证集最小的loss 或 最大f1-score或准确率
for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    # 开始训练
    for step, batch in enumerate(train_dataloader):
        # 梯度清零
        model.zero_grad()
        # 计算loss
        #loss, logits, hidden_states = model(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0), labels=batch[1])
        output = model(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0), labels=batch[1])
        loss = output.loss
        total_loss += loss.item()
        # 梯度回传
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 梯度更新
        optimizer.step()
        scheduler.step()
    # model.eval()表示模型切换到eval模式，表示不会更新参数，只有在train模式下，才会更新梯度参数
    model.eval()
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            #loss, logits, hidden_states = model(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0), labels=batch[1])
            #print(loss, logits)
            output = model(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0), labels=batch[1])
            loss = output.loss
            logits = output.logits
            total_val_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()

    avg_val_loss = total_val_loss / len(val_dataloader)
    if avg_val_loss < dev_best_loss:
        last_improve = total_batch

    if total_batch - last_improve > require_improvement:
        # 验证集loss超过1000batch没下降，结束训练
        print("No optimization for a long time, auto-stopping...")
        flag = True
        break
    if flag:
        break
    total_batch = total_batch + 1