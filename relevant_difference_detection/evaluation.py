import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def evaluate(model, test_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test, total_f1_test, total_precision_test, total_recall_test, total_auc_test = 0, 0, 0, 0, 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            #accuracy
            #acc = (output.argmax(dim=1) == test_label).sum().item()
            acc = accuracy_score(test_label.cpu(), output.argmax(dim=1).cpu())
            total_acc_test += acc

            #f1
            f1 = f1_score(test_label.cpu(), output.argmax(dim=1).cpu(), average='macro')
            total_f1_test += f1

            #precision
            precision = precision_score(test_label.cpu(), output.argmax(dim=1).cpu(), average='macro')
            total_precision_test += precision

            #recall
            recall = recall_score(test_label.cpu(), output.argmax(dim=1).cpu(), average='macro')
            total_recall_test += recall


    
    print(f'Test Accuracy: {total_acc_test / len(test_dataloader): .3f}')
    print(f'Test F1: {total_f1_test / len(test_dataloader): .3f}')
    print(f'Test Precision: {total_precision_test / len(test_dataloader): .3f}')
    print(f'Test Recall: {total_recall_test / len(test_dataloader): .3f}')
    
    