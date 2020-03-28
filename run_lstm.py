import torch
from model import LSTMModel
from data_class import SequentialDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from sklearn import metrics

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import auc

# set the following parameters
input_size = 8
output_size = 1
hidden_dim = 32
num_layers = 2
dropout = 0.1
batch_size = 1
num_epochs = 1

train_dataset = SequentialDataset(transactions_path='balanced_data_split/train_transactions.csv',
                                  targets_path='balanced_data_split/train_labels.csv')
# train_dataset = SequentialDataset(transactions_path='data/transactions_in_january_reduced.csv',
#                                   targets_path='data/february_labels_reduced.csv')
test_dataset = SequentialDataset(transactions_path='balanced_data_split/test_transactions.csv',
                                 targets_path='balanced_data_split/test_labels.csv')

# transactions = train_dataset.transactions
# users = train_dataset.users
# targets = train_dataset.targets
#
# percentage = targets['is_churn'].value_counts(normalize=True) * 100
# print("percentage = ", percentage)

train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True)

test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

model = LSTMModel(input_size=input_size,
                  output_size=output_size,
                  hidden_dim=hidden_dim,
                  num_layers=num_layers,
                  batch_size=batch_size,
                  dropout=dropout)

lr = 0.001
# loss function = binary cross entropy
criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

print_every = 1000
model = model.double()

model.train()

total_loss = 0.0
total_corrects = 0

for epoch in range(num_epochs):
    batch_loss = 0.0
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

        bce_loss = criterion(predicted, target)

        optimizer.zero_grad()

        bce_loss.backward()

        optimizer.step()

        batch_loss += bce_loss
        total_loss += bce_loss
        if i % print_every == 0 and i > 0:
            print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, batch_size / print_every))
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

torch.save(model.state_dict(), "trained_models/LSTM_model.pt")
print('Trained Model Saved')

print('\n\n Testing...')
model.eval()
total_test_loss = 0.0
for i, data in enumerate(test_data_loader, 0):
    print("data = ", data)
    inputs = data[0]
    target = data[1]

    # print("data size = ", len(data))
    print("inputs = ", inputs)
    print("shape = ", inputs.shape)
    print("target = ", target)
    print("shape target = ", target.shape)
    padded_inputs = nn.utils.rnn.pad_sequence(sequences=inputs,
                                              batch_first=True,
                                              padding_value=0)
    predicted, hidden = model(padded_inputs)

    bce_loss = criterion(predicted, target)
    total_test_loss += bce_loss

    if i % print_every == 0 and i > 0:
        print('[%5d] loss: %.7f' % (i + 1, total_test_loss / (i + 1)))
        print('real: ', str(target.item()), '----- predicted: ', str(predicted.item()))
        print()

print('Finished Testing')
print('\n Test Loss : ', total_test_loss / len(test_data_loader), '\n')
