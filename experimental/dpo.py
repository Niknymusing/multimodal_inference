import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=1)  # Softmax over the second dimension for batch processing
        )
        
    def forward(self, x):
        return self.layers(x)

def dpo_loss(πθ, πref, x, y_w, y_l, beta):
    πθ_output = πθ(x)
    πref_output = πref(x)
    # Indexing correctly considering batch dimension
    log_ratio_w = torch.log(πθ_output[:, y_w] / πref_output[:, y_w])
    log_ratio_l = torch.log(πθ_output[:, y_l] / πref_output[:, y_l])
    logit_preference = beta * (log_ratio_w - log_ratio_l)
    return -torch.log(torch.sigmoid(logit_preference))

# Dimensions and data
input_size = 10
output_size = 5  # Ensure this matches the number of actions

πθ = PolicyNetwork(input_size, output_size)
πref = PolicyNetwork(input_size, output_size)

# Example data with batch dimension
x = torch.rand(1, input_size)  # Batch size of 1
y_w = 2                        # Preferred action
y_l = 3                        # Less preferred action
y_w = torch.tensor([y_w])      # Ensure y_w, y_l are tensors
y_l = torch.tensor([y_l])
beta = 1.0

loss = dpo_loss(πθ, πref, x, y_w, y_l, beta)

optimizer = optim.Adam(πθ.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())