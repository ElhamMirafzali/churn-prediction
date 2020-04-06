from one_branch_LSTM.transactions_data_class import TransactionsSequentialDataset
from one_branch_LSTM.logs_data_class import LogsSequentialDataset
from one_branch_LSTM.model import OneBranchLSTMModel
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from sklearn import metrics

input_size = 7  # logs
# input_size = 33  # transactions
output_size = 1
hidden_dim_lstm = 32
num_layers_lstm = 2
dropout = 0.1
batch_size = 1

# transactions
# train_dataset = TransactionsSequentialDataset(transactions_path='../balanced_data_split/train_transactions.csv',
#                                               targets_path='../balanced_data_split/train_labels_selected.csv')
#
# test_dataset = TransactionsSequentialDataset(transactions_path='../balanced_data_split/test_transactions.csv',
#                                              targets_path='../balanced_data_split/test_labels.csv')

# logs
train_dataset = LogsSequentialDataset(logs_path='../balanced_data_split/train_logs.csv',
                                      targets_path='../balanced_data_split/train_labels_selected.csv')

test_dataset = LogsSequentialDataset(logs_path='../balanced_data_split/test_logs.csv',
                                     targets_path='../balanced_data_split/test_labels.csv')

train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True)

test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

model = OneBranchLSTMModel(input_size=input_size,
                           output_size=output_size,
                           hidden_dim=hidden_dim_lstm,
                           num_layers=num_layers_lstm,
                           batch_size=batch_size,
                           dropout=dropout).double()

num_epochs = 1
lr = 0.001
print_every = 1000

# loss function = binary cross entropy
criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# model = model.double()

model.train()

total_loss = 0.0
total_corrects = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    batch_corrects = 0
    targets = []
    predictions = []
    bin_predictions = []
    for i, (inputs, target) in enumerate(train_data_loader, 0):

        # inputs = data[0]
        # target = data[1]

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
        total_loss += bce_loss.item()
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

    total_corrects += batch_corrects
    print('\nEpoch %d/%d, Accuracy: %.3f' % (epoch + 1, num_epochs, batch_corrects / len(train_data_loader)))

    # calculate roc curve
    auc = metrics.roc_auc_score(targets, predictions)
    print('\nEpoch %d/%d, AUC: %.3f' % (epoch + 1, num_epochs, auc))

    # calculate precision-recall curve
    precision, recall, _ = metrics.precision_recall_curve(targets, predictions)
    pr_auc = metrics.auc(recall, precision)
    print('\nEpoch %d/%d, PR-AUC: %.3f' % (epoch + 1, num_epochs, pr_auc))

    # calculate precision
    average_precision = metrics.average_precision_score(targets, predictions)
    print('\nEpoch %d/%d, Precision: %.3f' % (epoch + 1, num_epochs, average_precision))

    # calculate recall
    average_recall = metrics.recall_score(targets, bin_predictions)
    print('\nEpoch %d/%d, Recall: %.3f' % (epoch + 1, num_epochs, average_recall))

print('Finished Training')
print('\nTrain Loss : %.7f' % (total_loss / len(train_data_loader)))

print('\nTrain Accuracy : %.3f' % (total_corrects / (len(train_data_loader) * num_epochs)), '\n')

# torch.save(model.state_dict(), "trained_models/LSTM_model.pt")
# print('Trained Model Saved')

# print('\n\n Testing...')
# model.eval()
# total_test_loss = 0.0
# for i, data in enumerate(test_data_loader, 0):
#     print("data = ", data)
#     inputs = data[0]
#     target = data[1]
#
#     # print("data size = ", len(data))
#     print("inputs = ", inputs)
#     print("shape = ", inputs.shape)
#     print("target = ", target)
#     print("shape target = ", target.shape)
#     padded_inputs = nn.utils.rnn.pad_sequence(sequences=inputs,
#                                               batch_first=True,
#                                               padding_value=0)
#     predicted, hidden = model(padded_inputs)
#
#     bce_loss = criterion(predicted, target)
#     total_test_loss += bce_loss
#
#     if i % print_every == 0 and i > 0:
#         print('[%5d] loss: %.7f' % (i + 1, total_test_loss / (i + 1)))
#         print('real: ', str(target.item()), '----- predicted: ', str(predicted.item()))
#         print()
#
# print('Finished Testing')
# print('\n Test Loss : ', total_test_loss / len(test_data_loader), '\n')
