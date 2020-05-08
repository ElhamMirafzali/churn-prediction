import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from sklearn import metrics
import statistics
from one_branch_LSTM.transactions_data_class import TransactionsSequentialDataset
from one_branch_LSTM.logs_data_class import LogsSequentialDataset
from one_branch_LSTM.model import OneBranchLSTMModel
from calculate_metrics import calculate_metrics
from preprocess import preprocess_logs
import pandas as pd

input_size = 7  # logs
# input_size = 30  # transactions
output_size = 1
hidden_dim_lstm = 32
num_layers_lstm = 2
fc1_units = 128
fc2_units = 32
dropout = 0.1
batch_size = 1

# transactions
# train_dataset = TransactionsSequentialDataset(transactions_path='../new_data/selected/train_transactions.csv',
#                                               targets_path='../new_data/selected/train_labels.csv')
#
# test_dataset = TransactionsSequentialDataset(transactions_path='../new_data/selected/test_transactions.csv',
#                                              targets_path='../new_data/selected/test_labels.csv')

# logs

train_logs = pd.read_csv('../new_data/selected2/train_logs_preprocessed.csv')
test_logs = pd.read_csv('../new_data/selected2/test_logs_preprocessed.csv')

train_dataset = LogsSequentialDataset(logs=train_logs,
                                      targets_path='../new_data/selected2/train_labels.csv')
test_dataset = LogsSequentialDataset(logs=test_logs,
                                     targets_path='../new_data/selected2/test_labels.csv')

train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True)

test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

model = OneBranchLSTMModel(input_size=input_size,
                           output_size=output_size,
                           hidden_dim=hidden_dim_lstm,
                           num_layers=num_layers_lstm,
                           fc1_units=fc1_units,
                           fc2_units=fc2_units,
                           batch_size=batch_size,
                           dropout=dropout).double()

num_epochs = 1
lr = 0.001
print_every = 1000

# loss function = binary cross entropy
criterion = nn.BCELoss()
# criterion = nn.NLLLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(params=model.parameters(), lr=lr)
# optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9, nesterov=True, weight_decay=0.01)
# optimizer = optim.RMSprop(params=model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.8, momentum=0.01)

model.train()

total_loss = []
total_corrects = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    batch_corrects = 0
    targets = []
    predictions = []
    bin_predictions = []
    # Training
    for i, (inputs, target) in enumerate(train_data_loader, 0):

        # print("shape of input for user %i th = " % i, inputs.shape)

        padded_inputs = nn.utils.rnn.pad_sequence(sequences=inputs,
                                                  batch_first=True,
                                                  padding_value=0)

        predicted, hidden = model(padded_inputs)
        # last_prediction = predicted1[:, -1, :]

        bce_loss = criterion(predicted, target)

        optimizer.zero_grad()

        bce_loss.backward()

        optimizer.step()

        running_loss += bce_loss.item()
        epoch_loss += bce_loss.item()
        if i % print_every == 0 and i > 0:
            print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / print_every))
            print('real: ', str(target.item()), '----- predicted: ', str(predicted.item()))
            running_loss = 0.0
            print()

        targets.append(target.item())
        predictions.append(predicted.item())
        # Accuracy
        predicted_label = (predicted >= 0.5).float()
        bin_predictions.append(predicted_label)
        if predicted_label == target:
            batch_corrects += 1

    total_loss.append((epoch_loss / len(train_data_loader)))

    total_corrects += batch_corrects
    auc, pr_auc, average_precision, average_recall = calculate_metrics(targets=targets, predictions=predictions,
                                                                       bin_predictions=bin_predictions)

    print('\nEpoch %d/%d, Accuracy: %.3f' % (epoch + 1, num_epochs, batch_corrects / len(train_data_loader)))
    print('\nEpoch %d/%d, Loss : %.7f' % (epoch + 1, num_epochs, (epoch_loss / len(train_data_loader))))
    print('\nEpoch %d/%d, AUC: %.3f' % (epoch + 1, num_epochs, auc))
    print('\nEpoch %d/%d, PR-AUC: %.3f' % (epoch + 1, num_epochs, pr_auc))
    print('\nEpoch %d/%d, Precision: %.3f' % (epoch + 1, num_epochs, average_precision))
    print('\nEpoch %d/%d, Recall: %.3f' % (epoch + 1, num_epochs, average_recall), '\n')

print('Finished Training')
print('\nTrain Loss : %.7f' % statistics.mean(total_loss))
print('\nTrain Accuracy : %.3f' % (total_corrects / (len(train_data_loader) * num_epochs)), '\n')

torch.save(model.state_dict(), "../trained_models/LSTM_model.pt")
print('Trained Model Saved')

print('\n\n Testing...')
model.load_state_dict(torch.load("../trained_models/LSTM_model.pt"))
model.eval()
total_test_loss = 0.0
total_corrects = 0
targets = []
predictions = []
bin_predictions = []

for i, (inputs, target) in enumerate(test_data_loader, 0):

    # padded_inputs = nn.utils.rnn.pad_sequence(sequences=inputs,
    #                                           batch_first=True,
    #                                           padding_value=0)
    predicted, hidden = model(inputs)

    bce_loss = criterion(predicted, target)
    total_test_loss += bce_loss

    if i % print_every == 0 and i > 0:
        print('[%5d] loss: %.7f' % (i + 1, total_test_loss / (i + 1)))
        print('real: ', str(target.item()), '----- predicted: ', str(predicted.item()))
        print()

    targets.append(target.item())
    predictions.append(predicted.item())
    # Accuracy
    predicted_label = (predicted >= 0.5).float()
    bin_predictions.append(predicted_label)
    if predicted_label == target:
        total_corrects += 1

auc, pr_auc, average_precision, average_recall = calculate_metrics(targets=targets, predictions=predictions,
                                                                   bin_predictions=bin_predictions)
print('Finished Testing')
print('\nAccuracy: %.3f' % (total_corrects / len(test_data_loader)))
print('\nAUC: %.3f' % auc)
print('\nPR-AUC: %.3f' % pr_auc)
print('\nPrecision: %.3f' % average_precision)
print('\nRecall: %.3f' % average_recall)
print('\nTest Loss : %.7f' % (total_test_loss / len(test_data_loader)), '\n')
