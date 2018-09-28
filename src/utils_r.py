from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np

def generate_validation_set(session, tracking, test_size=0.1, seed=42):
    '''
    trn_session, trn_tracking, val_session, val_tracking = generate_validation_set(train_session, 
                                                                train_tracking, test_size=0.1, seed=42)
    '''
    trn_session, val_session = train_test_split(session, test_size=test_size, random_state=seed, shuffle=True)
    trn_session_ids, val_session_ids = set(trn_session.sid), set(val_session.sid)
    trn_tracking = tracking[tracking.sid.apply(lambda x: x in trn_session_ids)]
    val_tracking = tracking[tracking.sid.apply(lambda x: x in val_session_ids)]
    return trn_session, trn_tracking, val_session, val_tracking

def add_num_skus(session, tracking):
    sku_counts = tracking.groupby('sid')['sku'].nunique()
    session['NUM_SKUS'] = session.sid.map(sku_counts)
    
class MultimodalDataset(Dataset):
    def __init__(self, cats, conts, cat_seqs, cont_seqs, targets=None):
        self.cats = np.array(cats).astype(np.int64)
        self.conts = np.array(conts).astype(np.float32)
        self.cat_seqs = np.array(cat_seqs).astype(np.int64)
        self.cont_seqs = np.array(cont_seqs).astype(np.float32)
        self.targets = np.array(targets).astype(np.float32) \
                            if targets is not None else \
                            np.zeros(len(cats)).astype(np.float32)
    
    def __len__(self):
        return len(self.cats)
    
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx],
                self.cat_seqs[idx], self.cont_seqs[idx], self.targets[idx]]

class StructuredNet(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops, out_sz):
        super().__init__()        
        self.embs = nn.ModuleList([
            nn.Embedding(c, s) for c,s in emb_szs
        ])
        for emb in self.embs:
            self.emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont
        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)
        ])
        for o in self.lins:
            nn.init.kaiming_normal_(o.weight.data)
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]
        ])
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([
            nn.Dropout(drop) for drop in drops
        ])
        self.bn = nn.BatchNorm1d(n_cont)
        
    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn(x_cont)
        x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for lin, drop, bn in zip(self.lins, self.drops, self.bns):
            x = F.relu(lin(x))
            x = bn(x)
            x = drop(x)
        return self.outp(x)
    
    def emb_init(self, x):
        x = x.weight.data
        sc = 2 / (x.size(1) + 1)
        x.uniform_(-sc, sc)    
    
class MultimodalNet(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops, 
                 rnn_hidden_sz, rnn_cont_sz, rnn_emb_sz, rnn_n_layers,
                 rnn_drop, out_sz=1):
        super().__init__()
        self.structured_net = StructuredNet(emb_szs, n_cont=n_cont, 
                        emb_drop=emb_drop, szs=szs, drops=drops, 
                        out_sz=rnn_hidden_sz)
        
        self.embs = nn.ModuleList([
            nn.Embedding(c, s) for c,s in rnn_emb_sz
        ])
        for emb in self.embs:
            self.emb_init(emb)
        
        n_emb = sum(e.embedding_dim for e in self.embs)
        rnn_input_sz = rnn_cont_sz + n_emb
        
        self.lstm = nn.LSTM(rnn_input_sz, rnn_hidden_sz, rnn_n_layers, 
                            dropout=rnn_drop)
        self.out = nn.Linear(rnn_hidden_sz * 2, out_sz) # [struct_out, rnn_out]
        
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_sz = rnn_hidden_sz
        
    def forward(self, cats, conts, cat_seqs, cont_seqs, hidden):
        # cont_seqs: [bs, inp, seq]
        # cat_seqs: [bs, inp, seq]
        metadata = self.structured_net(cats, conts) # [bs, hs]
        # cell = x.unsqueeze(0).repeat(self.rnn_n_layers, 1, 1) # [nlay, bs, hs]
        cell = metadata.unsqueeze(0).expand(self.rnn_n_layers, -1, 
                                            -1).contiguous()
        
        # DONE: cat_seqs embeddings
        cont_seqs = cont_seqs # .unsqueeze(2) 
        
        embedded_cat_seqs = [emb(cat_seqs[:,:,i]) for i, emb in \
                             enumerate(self.embs)]
        
        embedded_cat_seqs = torch.cat(embedded_cat_seqs, 2)
        # embedded_cat_seqs: (BS, seqlen, embedding sizes sum)
        
        seqs = torch.cat([cont_seqs, embedded_cat_seqs], 2)
        # seqs: [seqlen, bs, inp]
        # seqs = seqs.transpose(1,0).transpose(2,0)
        # seqs: [inp, bs, seqlen]
        
        seqs = seqs.transpose(1,0)
        
        # input: (seq_len, batch, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        # (2, 5, 15)
        outputs, hidden = self.lstm(seqs, (hidden, cell))
        out = self.out(torch.cat([metadata, outputs[-1]], 1)) 
        # [struct_out, rnn_out]
        return out
        
    def init_hidden(self, batch_sz):
        return torch.zeros(self.rnn_n_layers, batch_sz, self.rnn_hidden_sz)
    
    def emb_init(self, x):
        x = x.weight.data
        sc = 2 / (x.size(1) + 1)
        x.uniform_(-sc, sc)  
    
def train_step(model, cats, conts, cat_seqs, cont_seqs, hidden, 
               targets, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    preds = model(cats, conts, cat_seqs, cont_seqs, hidden)
    loss = criterion(preds.view(-1), targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_model(model, data_loader, USE_CUDA=False):
    targets, preds = [], [] 
    model.eval()
    for batch_idx, (cats, conts, cat_seqs, cont_seqs, target) in enumerate(data_loader):
        with torch.no_grad():
            hidden = model.init_hidden(len(cats))
            if USE_CUDA:
                cats, conts, cat_seqs, cont_seqs, hidden, target = cats.cuda(), \
                              conts.cuda(), cat_seqs.cuda(), \
                              cont_seqs.cuda(), hidden.cuda(), target.cuda()
            pred = model(cats, conts, cat_seqs, cont_seqs, hidden)
            targets.extend(target.cpu())
            preds.extend(pred.cpu())
            assert len(targets) == len(preds)
    return [x.item() for x in targets], [torch.sigmoid(x).item() for x in preds]

def get_metrics(model, data_loader, USE_CUDA=False):
    targets, preds = eval_model(model, data_loader, USE_CUDA=USE_CUDA)
    return log_loss(targets, preds)

def train_model(model, train_loader, val_loader, optimizer, criterion,
                n_epochs, USE_CUDA=False):
    val_losses = []
    for epoch in range(n_epochs):
        for batch_idx, (cats, conts, cat_seqs, cont_seqs, target) in enumerate(train_loader):
            hidden = model.init_hidden(len(cats))
            if USE_CUDA:
                cats, conts, cat_seqs, cont_seqs, hidden, target = cats.cuda(), \
                              conts.cuda(), cat_seqs.cuda(), \
                              cont_seqs.cuda(), hidden.cuda(), target.cuda()
            train_step(model, cats, conts, cat_seqs, cont_seqs,
                       hidden, target, optimizer, criterion)
        
        if val_loader is not None:
            train_loss = get_metrics(model, train_loader, USE_CUDA)
            val_loss = get_metrics(model, val_loader, USE_CUDA)
            print(f'Epoch: {epoch+1} | Train Logloss: {train_loss:.2f} | '
                  f'Val Logloss: {val_loss:.2f}')
            val_losses.append(val_loss)
            
        else:
            train_loss = get_metrics(model, train_loader, USE_CUDA)
            print(f'Epoch: {epoch+1} | Train Logloss: {train_loss:.2f}')
            
        torch.save(model.state_dict(), f'data/neuralnet/model_e{epoch}.pt')    
        
    return max(range(len(val_losses)), key=lambda x: val_losses[x]) \
            if val_loader is not None else n_epochs