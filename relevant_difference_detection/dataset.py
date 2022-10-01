import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer


datapath = '../generated_data/similar_sentences.xlsx'
df = pd.read_excel(datapath)
labels = {'RELEVANT': 0, 'STYLYSTIC': 1, 'IRRELEVANT': 2}

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len=512):

        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(s1, s2, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for s1, s2 in zip(df['sentence1'], df['sentence2'])]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def get_dataloader(df, batch_size=16, shuffle=True, tokenizer = 'bert-base-cased'):

    if tokenizer == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        max_len = 512
    
    np.random.seed(42)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                        [int(.8*len(df)), int(.9*len(df))])

    print('train:', len(df_train), 'val:', len(df_val), 'test:', len(df_test))

    train_dataset = Dataset(df_train, tokenizer, max_len)
    val_dataset = Dataset(df_val, tokenizer, max_len)
    test_dataset = Dataset(df_test, tokenizer, max_len)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader, test_dataloader