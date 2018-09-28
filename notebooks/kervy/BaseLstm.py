import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassification(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu=True, dropout=0.5):
        super(LSTMClassification, self).__init__()
        
        # Parameters
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        
        # Layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        
        # Initialization
        self.hidden = self.init_hidden()
        self.init_weights()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (torch.zeros(1, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, self.batch_size, self.hidden_dim),
                    torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        