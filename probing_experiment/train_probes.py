import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

activations_path = 'cached_activations/final/transformer_layer_5_first_pred.npy'
probe1_outputs_path = 'probe1_expected_outputs.npy'

activations = np.load(activations_path)  # Shape: (1000, 768)
outputs = np.load(probe1_outputs_path)   # Shape: (1000,)
## Hacky normalization
outputs_min = outputs.min()
outputs_max = outputs.max()
outputs_normalized = (outputs - outputs_min) / (outputs_max - outputs_min) * 100


X_train, X_test, y_train, y_test = train_test_split(
    activations, outputs_normalized, test_size=0.1, random_state=42
)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

y_train = y_train.unsqueeze(1).float()
y_test = y_test.unsqueeze(1).float()


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(768, 20)   
        self.output = nn.Linear(20, 1)     

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)  # Activation function
        x = self.output(x)
        return x

model = LinearModel()


criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)


initial_lr = 0.01

num_epochs = 10000
best_loss = float('inf')
plateau_count = 0
current_lr = initial_lr


train_losses = []
test_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs_pred = model(inputs)
        loss = criterion(outputs_pred, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if epoch % 100 == 0:
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs_pred = model(inputs)
                loss = criterion(outputs_pred, targets)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Check if loss has improved
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            plateau_count = 0
        else:
            plateau_count += 1
            
        if plateau_count >= 1:  # Since we're checking every 100 epochs, this means no improvement for 100 epochs
            current_lr = current_lr / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Learning rate reduced to {current_lr}")
            plateau_count = 0  # Reset counter
            
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {current_lr:.6f}")
