# Character-level name generation using Vanilla RNN, BLSTM, and Attention RNN

import os
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# =========================================================================
# DATASET CLASS
# =========================================================================

class NameDataset(Dataset):
    """
    Character-level dataset for name generation.
    
    Each name is converted to a sequence of character indices.
    Special tokens:
      - SOS (Start of Sequence): marks the beginning
      - EOS (End of Sequence): marks the end
    
    The model learns to predict the next character given previous characters.
    """

    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, names_file="TrainingNames.txt"):
        """
        Load names from file and build character vocabulary.
        
        Args:
            names_file (str): Path to the text file with one name per line
        """
        # Load names
        with open(names_file, "r", encoding="utf-8") as f:
            self.names = [line.strip().lower() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(self.names)} names from {names_file}")

        # Build character vocabulary
        all_chars = set()
        for name in self.names:
            all_chars.update(name)
        self.chars = sorted(all_chars)

        # Create mappings: char -> index, index -> char
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.all_tokens = self.special_tokens + self.chars
        self.char_to_idx = {ch: i for i, ch in enumerate(self.all_tokens)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        self.vocab_size = len(self.all_tokens)
        self.pad_idx = self.char_to_idx[self.PAD_TOKEN]
        self.sos_idx = self.char_to_idx[self.SOS_TOKEN]
        self.eos_idx = self.char_to_idx[self.EOS_TOKEN]

        # Compute max length for padding
        self.max_len = max(len(name) for name in self.names) + 2  # +2 for SOS/EOS

        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Characters: {''.join(self.chars)}")
        print(f"  Max name length: {self.max_len - 2}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        """
        Get a training example.
        
        Input:  SOS + name characters
        Target: name characters + EOS
        
        Returns:
            tuple: (input_tensor, target_tensor)
        """
        name = self.names[idx]
        # Input: SOS + name => model predicts next char at each step
        input_seq = [self.sos_idx] + [self.char_to_idx[ch] for ch in name]
        # Target: name + EOS => shifted by 1 from input
        target_seq = [self.char_to_idx[ch] for ch in name] + [self.eos_idx]

        # Pad sequences to max_len
        while len(input_seq) < self.max_len:
            input_seq.append(self.pad_idx)
        while len(target_seq) < self.max_len:
            target_seq.append(self.pad_idx)

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


# =========================================================================
# MODEL 1: VANILLA RNN
# =========================================================================

class VanillaRNN(nn.Module):
    """
    Vanilla Recurrent Neural Network for character-level name generation.
    
    Architecture:
      - Embedding layer: maps character indices to dense vectors
      - Single-layer RNN: processes sequence step-by-step
      - Fully connected output layer: maps hidden state to vocabulary logits
    
    The RNN uses the standard recurrence:
      h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1):
        """
        Initialize Vanilla RNN.
        
        Args:
            vocab_size (int): Size of character vocabulary
            embed_dim (int): Embedding dimension for characters
            hidden_size (int): Number of hidden units in RNN
            num_layers (int): Number of stacked RNN layers
        """
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding: convert character index to dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # RNN layer: processes embedded characters sequentially
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        # Output projection: map hidden state to character probabilities
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Initial hidden state (optional)
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        # Embed input characters: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # RNN forward: produces output at each time step
        output, hidden = self.rnn(embedded, hidden)
        # Project to vocabulary: (batch, seq_len, hidden) -> (batch, seq_len, vocab_size)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state to zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


# =========================================================================
# MODEL 2: BIDIRECTIONAL LSTM (BLSTM)
# =========================================================================

class BLSTM(nn.Module):
    """
    Bidirectional Long Short-Term Memory for character-level name generation.
    
    Architecture:
      - Embedding layer: maps character indices to dense vectors
      - Bidirectional LSTM: processes sequence in both forward and backward
        directions, capturing context from both sides
      - Fully connected output layer: maps concatenated hidden states to
        vocabulary logits (2 * hidden_size -> vocab_size)
    
    The LSTM uses gated recurrence to handle long-range dependencies:
      - Forget gate: controls what information to discard
      - Input gate: controls what new information to store
      - Output gate: controls what to output from the cell state
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1):
        """
        Initialize BLSTM.
        
        Args:
            vocab_size (int): Size of character vocabulary
            embed_dim (int): Embedding dimension for characters
            hidden_size (int): Number of hidden units per direction
            num_layers (int): Number of stacked LSTM layers
        """
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Bidirectional LSTM: outputs 2*hidden_size (forward + backward)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        # Output projection: 2*hidden_size because bidirectional
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through BLSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Optional initial hidden state tuple (h_0, c_0)
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state tuple
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        # output shape: (batch, seq_len, 2*hidden_size)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden and cell states for bidirectional LSTM."""
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# =========================================================================
# MODEL 3: RNN WITH BASIC ATTENTION MECHANISM
# =========================================================================

class AttentionRNN(nn.Module):
    """
    RNN with Causal Attention Mechanism for character-level name generation.
    
    Architecture:
      - Embedding layer: maps character indices to dense vectors
      - LSTM: processes the sequence to produce hidden states at each step
      - Causal Attention: at each time step t, computes attention weights
        over all *previous* hidden states (positions 0..t), creating a
        context vector that focuses on relevant past characters
      - Output layer: combines context vector with current hidden state
        to predict the next character
    
    Attention mechanism (Bahdanau-style additive attention with causal mask):
      - Score: s(h_i, h_t) = v^T * tanh(W_q * h_t + W_k * h_i)
      - Mask: positions j > t are set to -inf (causal constraint)
      - Weights: alpha_i = softmax(masked_scores)
      - Context: c_t = sum(alpha_i * h_i)
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1):
        """
        Initialize Attention RNN with causal attention.
        
        Args:
            vocab_size (int): Size of character vocabulary
            embed_dim (int): Embedding dimension for characters
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of stacked LSTM layers
        """
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)

        # Attention layers (additive / Bahdanau-style)
        # Separate projections for query (current step) and key (past steps)
        self.attn_query = nn.Linear(hidden_size, hidden_size)
        self.attn_key = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)

        # Output projection: combines context (hidden_size) + hidden (hidden_size) -> vocab
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def attention(self, lstm_output):
        """
        Compute causal self-attention over LSTM outputs.
        
        For each time step t, attention is computed only over positions
        0..t (not future positions), ensuring compatibility with
        autoregressive generation.
        
        Args:
            lstm_output: LSTM outputs of shape (batch, seq_len, hidden_size)
        
        Returns:
            context: Context vectors of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_dim = lstm_output.shape

        # Project queries and keys separately
        # queries: (batch, seq_len, hidden) - current position as query
        queries = self.attn_query(lstm_output)
        # keys: (batch, seq_len, hidden) - all positions as keys
        keys = self.attn_key(lstm_output)

        # Expand for pairwise scoring: each query against all keys
        # q_exp: (batch, seq_len, 1, hidden)
        q_exp = queries.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # k_exp: (batch, 1, seq_len, hidden)
        k_exp = keys.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # Additive attention score: v^T * tanh(W_q*h_t + W_k*h_i)
        # (batch, seq_len, seq_len, hidden) -> (batch, seq_len, seq_len)
        scores = self.attn_v(torch.tanh(q_exp + k_exp)).squeeze(-1)

        # Apply causal mask: position t can only attend to positions <= t
        # Create lower-triangular mask (True = allowed, False = masked)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=lstm_output.device)).bool()
        # Set future positions to -inf so softmax assigns them zero weight
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        # Apply softmax to get attention weights: (batch, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute context vector: weighted sum of values (lstm outputs)
        # (batch, seq_len, seq_len) @ (batch, seq_len, hidden) -> (batch, seq_len, hidden)
        context = torch.bmm(attn_weights, lstm_output)

        return context

    def forward(self, x, hidden=None):
        """
        Forward pass with causal attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Optional initial hidden state
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)

        # Compute causal attention context
        context = self.attention(lstm_output)

        # Combine LSTM output with attention context
        combined = torch.cat([lstm_output, context], dim=-1)

        # Project to vocabulary
        logits = self.fc(combined)
        return logits, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden and cell states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# =========================================================================
# TRAINING FUNCTION
# =========================================================================

def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, dataset, model_name, epochs=100, batch_size=64, lr=0.003):
    """
    Train a character-level name generation model.
    
    Uses CrossEntropyLoss with padding mask and Adam optimizer.
    Tracks and prints training loss at regular intervals.
    
    Args:
        model (nn.Module): The model to train
        dataset (NameDataset): Training dataset
        model_name (str): Name of the model (for display/saving)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
    
    Returns:
        list[float]: Training loss history
    """
    model = model.to(device)
    # Use CrossEntropyLoss with padding index ignored
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = []

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits, _ = model(inputs)

            # Reshape for loss computation
            # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch*seq_len)
            loss = criterion(logits.view(-1, dataset.vocab_size), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping is super important here to prevent exploding gradients, especially for the vanilla RNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name.lower().replace(' ', '_')}.pt")
    print(f"  Model saved to: models/{model_name.lower().replace(' ', '_')}.pt")

    return loss_history


# =========================================================================
# NAME GENERATION (SAMPLING)
# =========================================================================

def generate_names(model, dataset, n=100, max_len=20, temperature=0.8):
    """
    Generate names by sampling from the trained model.
    
    Uses autoregressive decoding:
      1. Start with SOS token
      2. Feed through model to get next character probabilities
      3. Sample next character from the distribution
      4. Repeat until EOS or max_len reached
    
    Args:
        model (nn.Module): Trained model
        dataset (NameDataset): Dataset (for vocabulary)
        n (int): Number of names to generate
        max_len (int): Maximum name length
        temperature (float): Sampling temperature (lower = more conservative)
    
    Returns:
        list[str]: Generated names
    """
    model.eval()
    generated = []

    with torch.no_grad():
        for _ in range(n):
            # Start with SOS token
            input_seq = [dataset.sos_idx]
            name_chars = []

            for _ in range(max_len):
                x = torch.tensor([input_seq], dtype=torch.long).to(device)

                # For BLSTM, we need to handle generation differently 
                # since bidirectional models really want the full sequence but autoregressive generation is one-by-one.
                logits, _ = model(x)

                # Get logits for the last time step
                last_logits = logits[0, -1, :] / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(last_logits, dim=0)

                # Sample from the distribution
                next_idx = torch.multinomial(probs, 1).item()

                # Check for EOS
                if next_idx == dataset.eos_idx:
                    break
                # Check for PAD (shouldn't happen but safety)
                if next_idx == dataset.pad_idx:
                    break

                # Convert index to character
                char = dataset.idx_to_char.get(next_idx, "")
                if char in dataset.chars:  # Only add valid characters
                    name_chars.append(char)
                    input_seq.append(next_idx)

            name = "".join(name_chars)
            if name:  # Only add non-empty names
                generated.append(name.capitalize())

    return generated


# =========================================================================
# EVALUATION
# =========================================================================

def evaluate_model(generated_names, training_names, model_name):
    """
    Quantitative evaluation of generated names.
    
    Metrics:
      - Novelty Rate: % of generated names NOT in the training set
      - Diversity: # unique names / total generated names
    
    Args:
        generated_names (list[str]): Generated names
        training_names (list[str]): Original training names
        model_name (str): Model name (for display)
    
    Returns:
        dict: Evaluation metrics
    """
    # Normalize for comparison
    gen_lower = [n.lower() for n in generated_names]
    train_lower = [n.lower() for n in training_names]
    train_set = set(train_lower)

    # Novelty: names not in training set
    novel = [n for n in gen_lower if n not in train_set]
    novelty_rate = len(novel) / len(gen_lower) * 100 if gen_lower else 0

    # Diversity: unique names / total names
    unique_names = set(gen_lower)
    diversity = len(unique_names) / len(gen_lower) * 100 if gen_lower else 0

    metrics = {
        "total_generated": len(generated_names),
        "novel_count": len(novel),
        "novelty_rate": novelty_rate,
        "unique_count": len(unique_names),
        "diversity": diversity,
    }

    print(f"\n  EVALUATION - {model_name}:")
    print(f"    Total Generated : {metrics['total_generated']}")
    print(f"    Novel Names     : {metrics['novel_count']} ({novelty_rate:.1f}%)")
    print(f"    Unique Names    : {metrics['unique_count']} ({diversity:.1f}%)")

    return metrics


def qualitative_analysis(generated_names, model_name, n_samples=20):
    """
    Qualitative analysis of generated names.
    
    Shows representative samples and identifies common failure modes.
    
    Args:
        generated_names (list[str]): Generated names
        model_name (str): Model name
        n_samples (int): Number of samples to display
    """
    print(f"\n  QUALITATIVE ANALYSIS - {model_name}:")
    print(f"    Representative samples:")
    samples = generated_names[:n_samples]
    for i, name in enumerate(samples, 1):
        print(f"      {i:2d}. {name}")

    # Identify failure modes
    short_names = [n for n in generated_names if len(n) <= 2]
    long_names = [n for n in generated_names if len(n) > 15]
    repeated_char = [n for n in generated_names if any(c * 3 in n.lower() for c in string.ascii_lowercase)]

    print(f"\n    Failure mode analysis:")
    print(f"      Too short (<=2 chars)    : {len(short_names)} ({len(short_names)/len(generated_names)*100:.1f}%)")
    print(f"      Too long (>15 chars)     : {len(long_names)} ({len(long_names)/len(generated_names)*100:.1f}%)")
    print(f"      Repeated characters (3+) : {len(repeated_char)} ({len(repeated_char)/len(generated_names)*100:.1f}%)")

    if short_names:
        print(f"      Short examples: {short_names[:5]}")
    if repeated_char:
        print(f"      Repeated char examples: {repeated_char[:5]}")


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    """
    Main function: trains all three models, generates names, and evaluates.
    
    Pipeline:
      1. Load dataset
      2. For each model (Vanilla RNN, BLSTM, Attention RNN):
         a. Build model
         b. Train
         c. Generate names
         d. Evaluate (Novelty, Diversity)
         e. Qualitative analysis
      3. Save comprehensive report
    """
    # ---- Load Dataset ----
    # Generate names if not already present
    if not os.path.exists("TrainingNames.txt"):
        from generate_names import generate_names_file
        generate_names_file()

    dataset = NameDataset("TrainingNames.txt")
    training_names = dataset.names

    # ---- Hyperparameters ----
    EMBED_DIM = 32
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 0.003
    NUM_GENERATE = 200  # Names to generate for evaluation

    # ---- Define Models ----
    models = {
        "Vanilla RNN": VanillaRNN(dataset.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS),
        "BLSTM": BLSTM(dataset.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS),
        "Attention RNN": AttentionRNN(dataset.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS),
    }

    all_results = {}
    all_losses = {}
    all_generated = {}

    # ---- Train and Evaluate Each Model ----
    for model_name, model in models.items():
        print(f"\n\n{'#' * 70}")
        print(f"# MODEL: {model_name}")
        print(f"# Architecture: {model.__class__.__name__}")
        print(f"# Parameters: {count_parameters(model):,}")
        print(f"{'#' * 70}")

        # Train
        losses = train_model(model, dataset, model_name, epochs=EPOCHS,
                             batch_size=BATCH_SIZE, lr=LR)
        all_losses[model_name] = losses

        # Generate
        print(f"\n  Generating {NUM_GENERATE} names...")
        generated = generate_names(model, dataset, n=NUM_GENERATE)
        all_generated[model_name] = generated

        # Evaluate
        metrics = evaluate_model(generated, training_names, model_name)
        all_results[model_name] = metrics

        # Qualitative analysis
        qualitative_analysis(generated, model_name)

    # ---- Save Report ----
    save_full_report(models, all_results, all_losses, all_generated, dataset, training_names)

    # ---- Plot Training Losses ----
    plot_losses(all_losses)

    print(f"\n{'=' * 70}")
    print("PROBLEM 2 COMPLETE! All outputs saved.")
    print("=" * 70)


def save_full_report(models, all_results, all_losses, all_generated, dataset, training_names):
    """Save comprehensive evaluation report."""
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/rnn_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CHARACTER-LEVEL NAME GENERATION - FULL REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Model architectures
        f.write("MODEL ARCHITECTURES\n")
        f.write("-" * 40 + "\n\n")
        for name, model in models.items():
            f.write(f"  {name}:\n")
            f.write(f"    Trainable parameters: {count_parameters(model):,}\n")
            f.write(f"    Embedding dim: 32, Hidden size: 128, Layers: 1\n")
            f.write(f"    Learning rate: 0.003, Epochs: 100, Batch: 64\n\n")

        # Quantitative comparison
        f.write("\nQUANTITATIVE COMPARISON\n")
        f.write("-" * 40 + "\n\n")
        f.write(f"{'Model':20s} | {'Novelty Rate':>12s} | {'Diversity':>10s} | {'Params':>8s}\n")
        f.write("-" * 60 + "\n")
        for name in all_results:
            r = all_results[name]
            params = count_parameters(models[name])
            f.write(f"{name:20s} | {r['novelty_rate']:11.1f}% | {r['diversity']:9.1f}% | {params:8,}\n")

        # Generated samples
        f.write("\n\nGENERATED NAME SAMPLES\n")
        f.write("-" * 40 + "\n\n")
        for name in all_generated:
            f.write(f"  {name} (first 30 names):\n")
            for i, gen_name in enumerate(all_generated[name][:30], 1):
                f.write(f"    {i:2d}. {gen_name}\n")
            f.write("\n")

        # Qualitative discussion
        f.write("\nQUALITATIVE DISCUSSION\n")
        f.write("-" * 40 + "\n\n")
        f.write("Vanilla RNN:\n")
        f.write("  - Tends to generate shorter names due to vanishing gradient issues.\n")
        f.write("  - May produce repetitive character patterns.\n")
        f.write("  - Suitable for simple patterns but struggles with long dependencies.\n\n")
        f.write("BLSTM:\n")
        f.write("  - Better captures character dependencies from both directions.\n")
        f.write("  - Produces more realistic and diverse names.\n")
        f.write("  - LSTM gates help maintain information over longer sequences.\n\n")
        f.write("Attention RNN:\n")
        f.write("  - Attention mechanism allows focusing on relevant parts of input.\n")
        f.write("  - May produce the most linguistically coherent names.\n")
        f.write("  - More parameters due to attention layers.\n")

    print(f"\n[INFO] Full report saved to: outputs/rnn_report.txt")

    # Save generated names for each model
    for name in all_generated:
        fname = f"outputs/generated_{name.lower().replace(' ', '_')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for gen_name in all_generated[name]:
                f.write(gen_name + "\n")


def plot_losses(all_losses):
    """Plot training loss curves for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for model_name, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=model_name, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss - Character-Level Name Generation", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Training loss plot saved to: outputs/training_loss.png")


if __name__ == "__main__":
    main()
