import torch
from model import LSTMModel
from data_class import SequentialDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

# set the following parameters
input_size = 8
output_size = 1
hidden_dim = 1
num_layers = 1
dropout = 0.1
batch_size = 1
epochs = 1

sequential_dataset = SequentialDataset(transactions_path='data/transactions_in_january_reduced.csv',
                                       targets_path='data/february_labels_reduced.csv')
transactions = sequential_dataset.transactions
users = sequential_dataset.users
targets = sequential_dataset.targets
print(targets)
percentage = targets['is_churn'].value_counts(normalize=True) * 100
print("percentage = ", percentage)

train_data_loader: data.DataLoader = data.DataLoader(dataset=sequential_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False)

model = LSTMModel(input_size=input_size,
                  output_size=output_size,
                  hidden_dim=hidden_dim,
                  num_layers=num_layers,
                  batch_size=batch_size,
                  dropout=dropout)

lr = 0.001
criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

print_every = 1000
model = model.double()
model.train()

for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(train_data_loader, 0):
        inputs = data[0]
        target = data[1]

        output, hidden = model(inputs)

        bce_loss = criterion(output, target)
        bce_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += bce_loss
        if i % print_every == 0 and i > 0:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / print_every))
            print('real: ', str(target.item()), '----- predicted: ', str(output.item()))
            running_loss = 0.0
            print()

print('Finished Training')
torch.save(model.state_dict(), "trained_models/LSTM_model.pt")
print('Trained Model Saved')
