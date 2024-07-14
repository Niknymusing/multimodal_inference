from torch import nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1024, num_layers=1):
        super(LSTMClassifier, self).__init__()
        # Initialize the LSTM
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)  # Ensures input and output tensors are of shape (batch, seq, feature)
        
        # Initialize the fully connected layer to map the LSTM outputs to logits
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out, (hidden, cell) = self.lstm(x)
        
        # We use the last hidden state to classify. 
        # out[:, -1, :] gives us the last layer hidden state of shape (batch_size, hidden_dim)
        out = self.fc(out[:, -1, :])
        
        return out
    

class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()
        self.ff1 = nn.LSTM(1, 64, 256, 512)
        self.ff2 = nn.Linear(512, 1024)
        self.ff3 = nn.Linear(1024, 2048)
        
    
    def forward(self, x):
        x = F.relu_(self.ff1(x))
        x = F.relu_(self.ff2(x))
        x = F.relu_(self.ff3(x))
        
        return x