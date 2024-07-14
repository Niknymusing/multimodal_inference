import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Assume lθ comes from a language model, here we mock it as a simple neural network for demonstration
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits

def calculate_v_star(logits, beta):
    # logits: (batch_size, sequence_length, vocab_size)
    max_logit, _ = torch.max(logits, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    # Subtracting max_logit for numerical stability
    log_probs = logits - max_logit  # Broadcasting max_logit
    v_star = beta * torch.log(torch.sum(torch.exp(log_probs / beta), dim=-1, keepdim=True))  # Keepdim for shape match
    return v_star.squeeze(-1), log_probs  # Squeeze last dim for correct subtraction broadcasting

def token_level_dpo_loss(logits, actions, next_states, beta, pi_ref, terminal_mask):
    # Calculate V*
    v_star, log_probs = calculate_v_star(logits, beta)
    
    # Q* calculation for all actions, adjusted for broadcasting
    q_values = log_probs - v_star.unsqueeze(-1)  # Unsqueezing v_star for broadcasting over vocab_size dimension

    # Reference log-probabilities
    log_pi_ref = torch.log(pi_ref.gather(-1, actions.unsqueeze(-1)).squeeze(-1))

    # Calculate Q* for taken actions
    q_star_actions = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    # Adjust Q* for non-terminal states
    future_v_star = calculate_v_star(next_states, beta)[0]
    q_star_actions[~terminal_mask] += future_v_star[~terminal_mask]

    # Preference probabilities computation
    preferences = beta * (q_star_actions + log_pi_ref)

    return -torch.mean(torch.log(torch.sigmoid(preferences)))

# Other code remains the same for setup and usage


# Example usage
vocab_size = 100
embedding_dim = 50
hidden_dim = 100
beta = 0.1
lm = LanguageModel(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(lm.parameters(), lr=0.001)

# Mock input, actions, next_states, and pi_ref
input_seqs = torch.randint(0, vocab_size, (10, 5))
actions = torch.randint(0, vocab_size, (10, 5))
next_states = torch.randn(10, 5, vocab_size)  # Simulated next state logits
pi_ref = torch.full((10, 5, vocab_size), 0.1)  # Uniform reference policy for simplicity
terminal_mask = torch.zeros(10, 5, dtype=torch.bool)  # All non-terminal for simplicity

# Forward pass to get logits
logits = lm(input_seqs)

# Compute loss
loss = token_level_dpo_loss(logits, actions, next_states, beta, pi_ref, terminal_mask)

# Optimization step
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())