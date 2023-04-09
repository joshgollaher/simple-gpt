import torch
import torch.nn as nn
import torchtext
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import math

# Force enable cpu even when cuda is available
cpu_override = False

# Whether to train or test
train_mode = True


class GPTBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super(GPTBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)

        return x


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len):
        super(SimpleGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

        # Create positional encoding
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, max_seq_len, d_model)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pos_enc)

    def forward(self, x):
        # Generate positional encoding based on the input sequence length
        pos_enc = self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.embedding(x) + pos_enc
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

    def generate(self, input_tensor, num_tokens_to_generate=50):
        with torch.no_grad():
            generated_tokens = []
            for _ in range(num_tokens_to_generate):
                logits = self(input_tensor)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
                generated_tokens.append(next_token)
                input_tensor = torch.cat((input_tensor, next_token), dim=1)
            return torch.cat(generated_tokens, dim=1)


# Hyperparameters
vocab_size = 50000
d_model = 128
nhead = 4
d_ff = 3072
num_layers = 3
max_seq_len = 128


def tokenizer(x):
    return x.split()


def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)[:max_seq_len - 1]] + [vocab["<pad>"]]


def load_wikitext2(device, batch_size, max_seq_len):
    train_dataset, valid_dataset, test_dataset = torchtext.datasets.WikiText2()
    train_dataset = to_map_style_dataset(train_dataset)
    valid_dataset = to_map_style_dataset(valid_dataset)
    test_dataset = to_map_style_dataset(test_dataset)

    def tokenize(text):
        return text.strip().split()

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenize(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])

    pad_idx = vocab['<pad>']

    def collate_batch(batch):
        text_list = [torch.tensor(text_pipeline(text), dtype=torch.int64) for text in batch]
        text_lengths = [len(text) for text in text_list]
        text_list = nn.utils.rnn.pad_sequence(text_list, padding_value=pad_idx, batch_first=True)
        targets = torch.cat([text_list[:, 1:], pad_idx * torch.ones(text_list.size(0), 1, dtype=torch.int64)], dim=1)
        return text_list.to(device), targets.to(device)

    generator = torch.Generator(device='cpu')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True,
                                  generator=generator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_batch, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, generator=generator)

    return train_dataloader, valid_dataloader, test_dataloader, vocab


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    num_items = 2290  # Don't ask
    curr_iter = 0
    batch = 0
    for batch, (data, targets) in enumerate(dataloader):
        if curr_iter % 10 == 0:
            print(f'{(curr_iter / num_items * 100):.2f}% done')
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        curr_iter += 1

    return total_loss / (batch + 1)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, (data, targets) in enumerate(dataloader):
            logits = model(data)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    return total_loss / (batch + 1)


num_epochs = 5
batch_size = 16
learning_rate = 1e-4

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() and not cpu_override else 'cpu')

# Load the dataset
train_dataloader, valid_dataloader, test_dataloader, vocab = load_wikitext2(device, batch_size, max_seq_len)

# Update the vocab_size based on the loaded dataset
vocab_size = len(vocab)

# Initialize the model and send it to the device
model = SimpleGPT(vocab_size, d_model, nhead, num_layers, max_seq_len).to(device)

total_params = sum(
    param.numel() for param in model.parameters()
)

print(f'Model has {total_params} parameters.')

from pathlib import Path

if Path('model.ckpt').is_file():
    print(f'Loading model')
    model.load_state_dict(torch.load("model.ckpt"))

# Set up the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if train_mode:
    # Training loop
    for epoch in range(num_epochs):
        print(f'Training #{epoch + 1}')
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
        valid_loss = evaluate(model, valid_dataloader, loss_fn, device)

        print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
        torch.save(model.state_dict(), "model.ckpt")
else:
    def generate_text(model, input_text, text_pipeline, num_tokens_to_generate=50):
        model.eval()
        input_tokens = text_pipeline(input_text)
        input_tensor = torch.tensor(input_tokens, dtype=torch.int64).unsqueeze(0).to(device)
        output_tokens = model.generate(input_tensor, num_tokens_to_generate)
        output_tokens = output_tokens[0].cpu().numpy()
        output_text = ' '.join(
            [vocab.get_itos()[token] for token in output_tokens if vocab.get_itos()[token] != '<pad>'])
        return output_text


    def tokenize(text):
        return text.strip().split()


    print("Enter 'q' to quit.")
    while True:
        input_text = input("Enter text: ")
        if input_text.lower() == 'q':
            sys.exit(0)
        output_text = generate_text(model, input_text, text_pipeline)
        print(f"Generated text: {output_text}")