import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import pandas as pd
from two_branches_LSTM.model import TwoBranchesLSTMModel
from two_branches_LSTM.data_class import SequentialDataset
from calculate_metrics import calculate_metrics

input_size_transactions = 32
input_size_logs = 7
input_size_members = 41
output_size = 1
hidden_dim_lstm = 32
fc0_units = 32
fc1_units = 128
fc2_units = 32
# num_layers_lstm = 1
num_layers_lstm = 2
dropout = 0.1
batch_size = 1

# transactions
# train_transactions = pd.read_csv(
#     'new_data/selected2/new_normalization/train_transactions_preprocessed.csv')
test_transactions = pd.read_csv(
    'new_data/selected2/new_normalization/test_transactions_preprocessed.csv')

# logs
# train_logs = pd.read_csv('new_data/selected2/new_normalization/train_logs_preprocessed.csv')
test_logs = pd.read_csv('new_data/selected2/new_normalization/test_logs_preprocessed.csv')

# members
# train_members = pd.read_csv('new_data/selected2/train_members.csv')
test_members = pd.read_csv('new_data/selected2/test_members.csv')

# targets
# train_targets = pd.read_csv('new_data/selected2/train_labels.csv')
test_targets = pd.read_csv('new_data/selected2/test_labels.csv')

# train_dataset = SequentialDataset(transactions=train_transactions,
#                                   logs=train_logs,
#                                   members=train_members,
#                                   targets=train_targets)

test_dataset = SequentialDataset(transactions=test_transactions,
                                 logs=test_logs,
                                 members=test_members,
                                 targets=test_targets.loc[0:9000, :])

# train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset,
#                                                      batch_size=batch_size,
#                                                      shuffle=True)

test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

model = TwoBranchesLSTMModel(input_size_x1=input_size_transactions,
                             input_size_x2=input_size_logs,
                             input_size_x3=input_size_members,
                             output_size=output_size,
                             hidden_dim_lstm=hidden_dim_lstm,
                             num_layers=num_layers_lstm,
                             fc0_units=fc0_units,
                             fc1_units=fc1_units,
                             fc2_units=fc2_units,
                             batch_size=batch_size,
                             dropout=dropout).double()

num_epochs = 1
lr = 0.001
print_every = 1000

# loss function = binary cross entropy
criterion = nn.BCELoss()
# optimizer = optim.Adam(params=model.parameters(), lr=lr)
#
# model.train()
#
# total_loss = []
# total_corrects = 0
#
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     epoch_loss = 0.0
#     batch_corrects = 0
#     targets = []
#     predictions = []
#     bin_predictions = []
#     for i, (trans_data, logs_data, members_data, target) in enumerate(train_data_loader, 0):
#
#         padded_inputs1 = nn.utils.rnn.pad_sequence(sequences=trans_data,
#                                                    batch_first=True,
#                                                    padding_value=0)
#
#         padded_inputs2 = nn.utils.rnn.pad_sequence(sequences=logs_data,
#                                                    batch_first=True,
#                                                    padding_value=0)
#
#         padded_inputs3 = nn.utils.rnn.pad_sequence(sequences=members_data,
#                                                    batch_first=True,
#                                                    padding_value=0)
#
#         predicted = model(padded_inputs1, padded_inputs2, padded_inputs3)
#
#         bce_loss = criterion(predicted, target.view(-1, 1))
#
#         optimizer.zero_grad()
#
#         bce_loss.backward()
#
#         optimizer.step()
#
#         running_loss += bce_loss.item()
#         epoch_loss += bce_loss.item()
#
#         targets.append(target.item())
#         predictions.append(predicted.item())
#
#         # Accuracy
#         predicted_label = float(predicted.item() >= 0.5)
#         bin_predictions.append(predicted_label)
#         if predicted_label == target:
#             batch_corrects += 1
#
#         if i % print_every == 0 and i > 0:
#             print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / print_every))
#             print('real: ', str(target.item()), '----- predicted: ', str(predicted.item()))
#             running_loss = 0.0
#             print()
#
#     total_loss.append((epoch_loss / len(train_data_loader)))
#
#     total_corrects += batch_corrects
#     auc, pr_auc, average_precision, average_recall = calculate_metrics(targets=targets, predictions=predictions,
#                                                                        bin_predictions=bin_predictions)
#
#     print('\nEpoch %d/%d, Accuracy: %.3f' % (epoch + 1, num_epochs, batch_corrects / len(train_data_loader)))
#     print('\nEpoch %d/%d, Loss : %.7f' % (epoch + 1, num_epochs, (epoch_loss / len(train_data_loader))))
#     print('\nEpoch %d/%d, AUC: %.3f' % (epoch + 1, num_epochs, auc))
#     print('\nEpoch %d/%d, PR-AUC: %.3f' % (epoch + 1, num_epochs, pr_auc))
#     print('\nEpoch %d/%d, Precision: %.3f' % (epoch + 1, num_epochs, average_precision))
#     print('\nEpoch %d/%d, Recall: %.3f' % (epoch + 1, num_epochs, average_recall), '\n')
#
# print('Finished Training')
# print('\nTrain Loss : %.7f' % statistics.mean(total_loss))
# print('\nTrain Accuracy : %.3f' % (total_corrects / (len(train_data_loader) * num_epochs)), '\n')
#
# torch.save(model.state_dict(), "trained_models/LSTM_model.pt")
# print('Trained Model Saved')
#
print('\n Testing...')
model.load_state_dict(torch.load("trained_models/LSTM_model.pt"))
model.eval()
total_test_loss = 0.0
total_corrects = 0
targets = []
predictions = []
bin_predictions = []
running_loss = 0.0
for i, (trans_data, logs_data, members_data, target) in enumerate(test_data_loader, 0):

    padded_inputs1 = nn.utils.rnn.pad_sequence(sequences=trans_data,
                                               batch_first=True,
                                               padding_value=0)

    padded_inputs2 = nn.utils.rnn.pad_sequence(sequences=logs_data,
                                               batch_first=True,
                                               padding_value=0)

    padded_inputs3 = nn.utils.rnn.pad_sequence(sequences=members_data,
                                               batch_first=True,
                                               padding_value=0)

    predicted = model(padded_inputs1, padded_inputs2, padded_inputs3)

    bce_loss = criterion(predicted, target.view(-1, 1))
    total_test_loss += bce_loss
    running_loss += bce_loss

    if i % print_every == 0 and i > 0:
        print('[%5d] loss: %.7f' % (i + 1, running_loss / print_every))
        print('real: ', str(target.item()), '----- predicted: ', str(predicted.item()))
        running_loss = 0.0
        print()
    targets.append(target.item())
    predictions.append(predicted.item())
    # Accuracy
    predicted_label = float(predicted.item() >= 0.5)
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
print('\nTest Loss : %.7f' % (total_test_loss / len(test_data_loader)))
