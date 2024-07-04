import os
import subprocess
import sys


# Install portalocker if not installed
try:
    import portalocker  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "portalocker==2.10.0"])

# Install nltk if not installed
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    import evaluate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
    import evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Check and download spaCy models if necessary
import spacy.cli
try:
    spacy_ger = spacy.load("de_core_news_sm")
except OSError:
    spacy.cli.download("de_core_news_sm")
    spacy_ger = spacy.load("de_core_news_sm")

try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    spacy_eng = spacy.load("en_core_web_sm")

# Fonctions de tokenisation
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Building vocabularies
def yield_tokens(data_iter, tokenizer):
    for _, (src, trg) in enumerate(data_iter):
        yield tokenizer(src)
        yield tokenizer(trg)

train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=('de', 'en'))

vocab_ger = build_vocab_from_iterator(yield_tokens(train_data, tokenizer_ger), specials=["<sos>", "<eos>", "<pad>", "<unk>"], min_freq=1)
vocab_eng = build_vocab_from_iterator(yield_tokens(train_data, tokenizer_eng), specials=["<sos>", "<eos>", "<pad>", "<unk>"], min_freq=1)

print(f'Size of German vocabulary: {len(vocab_ger)}')
print(f'Size of English vocabulary: {len(vocab_eng)}')

# Define a default index for unknown tokens
vocab_ger.set_default_index(vocab_ger["<unk>"])
vocab_eng.set_default_index(vocab_eng["<unk>"])

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size=512, hidden_size=512, num_layers=3, p=0.3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p if num_layers > 1 else 0)
        
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.fc_cell = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.rnn(embedding)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.fc_hidden(hidden)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        cell = self.fc_cell(cell)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size=512, hidden_size=512, output_size=None, num_layers=3, p=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout=p if num_layers > 1 else 0)
        
        self.energy = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, 1)
        )
        
        self.softmax = nn.Softmax(dim=0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        sequence_length = encoder_states.shape[0]
        
        h_reshaped = hidden[-1].unsqueeze(0).repeat(sequence_length, 1, 1)
        
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(vocab_eng)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs

# Training parameters
learning_rate_initial = 0.01
learning_rate_final = 0.0005
learning_rate_decay = 0.9
learning_rate = learning_rate_initial
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using {device}')

input_size_encoder = len(vocab_ger)
input_size_decoder = len(vocab_eng)
output_size = len(vocab_eng)
encoder_embedding_size = 512
decoder_embedding_size = 512
hidden_size = 512
num_layers = 3
enc_dropout = 0.3
dec_dropout = 0.3

# Configuring TensorBoard
writer = SummaryWriter(f"runs/seq2seq_experiment")
step = 0

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_indices = [vocab_ger[token] if token in vocab_ger else vocab_ger['<unk>'] for token in tokenizer_ger(src_sample)]
        trg_indices = [vocab_eng[token] if token in vocab_eng else vocab_eng['<unk>'] for token in tokenizer_eng(trg_sample)]
        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_indices, dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=vocab_ger['<pad>'])
    trg_batch = pad_sequence(trg_batch, padding_value=vocab_eng['<pad>'])
    return src_batch, trg_batch

# Creating DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model instantiation
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = vocab_eng['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Load the model if it exists
start_from_checkpoint = False
model_path = '/kaggle/input/my-seq2seq-german2english-translator/pytorch/continue_training_from_checkpoint/1/german2english.pth'

if os.path.isfile(model_path):
    print("Loading model from checkpoint...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    print(f'Model loaded from {model_path} with step {step}')
else:
    print("No model found")
    if start_from_checkpoint:
        print("Exiting...")
        sys.exit()
    print("Starting from scratch...")

# Translation function
def translate_sentence(model, sentence, tokenizer_ger, vocab_ger, vocab_eng, device, max_length=50):
    tokens = ['<sos>'] + tokenizer_ger(sentence) + ['<eos>']
    text_to_indices = [vocab_ger[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)
    outputs = [vocab_eng['<sos>']]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
        best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        if output.argmax(1).item() == vocab_eng['<eos>']:
            break
    translated_sentence = [vocab_eng.lookup_token(idx) for idx in outputs]
    return translated_sentence


# Training loop
early_stopping_patience = 5
best_batch_loss = float('inf')
patience_counter = 0
epoch = 0
max_epochs = 100

for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

print("Starting training...")
print(f'Parameters are : initial learning_rate={learning_rate_initial}, final learning_rate={learning_rate_final}')
print(f'max_epochs={max_epochs}, patience={early_stopping_patience}')
while learning_rate >= learning_rate_final:
    model.train()
    train_loss = 0
    num_batches = 0
    for batch_idx, (src, trg) in enumerate(train_loader):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        train_loss += loss.item()
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1
        num_batches += 1
    
    train_loss = train_loss / num_batches

    valid_loss = 0
    num_batches = 0
    model.eval()
    for batch_idx, (src, trg) in enumerate(valid_loader):
        loss = criterion(output, trg)
        valid_loss += loss.item()
        writer.add_scalar("Validation loss", loss.item(), global_step=step)
        num_batches += 1
    
    train_loss = train_loss / num_batches
    validation_loss = valid_loss / num_batches


    # Early stopping
    if validation_loss < best_batch_loss:
        best_batch_loss = validation_loss
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, 'german2english.pth')
        print(f'=>Model saved')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            learning_rate = learning_rate * learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    if epoch >= max_epochs:
        break

    sentence = "zwei jungen stehen neben einem holzhaufen"
    translated_sentence = translate_sentence(model, sentence, tokenizer_ger, vocab_ger, vocab_eng, device, max_length=50)
    
    print(f'Epoch {epoch + 1} ; Learning rate: {param_group["lr"]} ; Training loss: {train_loss / num_batches} ; Validation loss: {validation_loss / num_batches} ; Patience: {patience_counter}')
    print(f'Translated Sentence: {" ".join(translated_sentence[1:-1])}')
    epoch += 1

# Final assessment on the test set with the best model
checkpoint = torch.load('german2english.pth')
model.load_state_dict(checkpoint['model_state_dict'])

#bleu = evaluate.load("bleu")
#test_bleu = calculate_bleu_score(model, test_loader, tokenizer_ger, vocab_ger, vocab_eng, device)
#print(f'Test BLEU: {test_bleu*100:.2f}')
