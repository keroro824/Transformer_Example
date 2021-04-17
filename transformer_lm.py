# %%
# +
import math
import torch
import transformer
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter
from transformer import TransformerEncoder, TransformerEncoderLayer
import time
import os
import numpy as np
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("This code requires APEX")

    
    

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, share_qk=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


# %%


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# %%
def batchify(data, bsz,device_):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device_)


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=20)
parser.add_argument("--test_size", type=int, default=10)
parser.add_argument("--dataset", type=str, default="wiki2")
parser.add_argument("--seq_lens", type=int, default=72)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--lr", type=float, default=5.0)
parser.add_argument("--emsize", type=int, default=200)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--nhid", type=int, default=200)
parser.add_argument("--local_rank", default=0, type=int, help="don't set this")
parser.add_argument('--log', action='store_true')
args = parser.parse_args()


args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = 0
args.world_size = 1

if args.distributed:
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    # args.gpu = args.local_rank
    # torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl')
    args.world_size = torch.distributed.get_world_size()


TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
if args.dataset=="wiki2": #WikiText103
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
elif args.dataset=="wiki103": #WikiText103
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText103.splits(TEXT)
    TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG=args.log
if LOG:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    writer = SummaryWriter("runs/"+args.dataset+"_lr_"+str(args.lr)+"_epochs_"+str(args.epochs)+"_seq_lens_"+str(args.seq_lens)+"_num_layers_"+str(args.num_layers)\
                      +"_num_heads_"+str(args.num_heads)+"_emsize_"+str(args.emsize)+"_nhid_"+str(args.nhid))
batch_size = args.batch_size
eval_batch_size = args.test_size  
epochs=args.epochs
train_data = batchify(train_txt, batch_size,device)
val_data = batchify(val_txt, eval_batch_size,device)
test_data = batchify(test_txt, eval_batch_size,device)


# %%

val_data=val_data[:350]
bptt = args.seq_lens
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


# %%


ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = args.emsize # embedding dimension
nhid = args.nhid # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = args.num_layers # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = args.num_heads # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
num_batches=int(train_data.size(0)/bptt)
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


# %%

GRADIENT_ACCUMULATE_EVERY = 4
criterion = nn.CrossEntropyLoss()
lr = args.lr # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
model, optimizer = amp.initialize(model, optimizer, opt_level='O0')


def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        
        output = model(data)
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = criterion(output.view(-1, ntokens), targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        

        total_loss += loss.item()
        log_interval = 50
#         print(batch)
        if batch % log_interval == 0 and batch!=0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if args.local_rank == 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // bptt, lr,
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                if LOG:
                    writer.add_scalar('train_ppl', math.exp(cur_loss), (epoch-1)*num_batches+batch)
            total_loss = 0
            start_time = time.time()
            
            val_loss,eig_scores,rank_scores,hopfield_scores = evaluate(model, val_data)
#             print(eig_scores)
            if LOG and args.local_rank == 0:
                for l in range(nlayers):
                    writer.add_scalar('train_eigen_score'+str(l), eig_scores[l], (epoch-1)*num_batches+batch)
                    writer.add_scalar('train_rank'+str(l), rank_scores[l], (epoch-1)*num_batches+batch)
                    writer.add_scalar('train_hopfield'+str(l), hopfield_scores[l], (epoch-1)*num_batches+batch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
            
                    
#             for l in range(nlayers):
#                 attn_tensor=model.transformer_encoder.layers[l].self_attn.multihead_attention.attn_output_weights.clone().detach()
#                 tmp_eig,avg_rank=eigen_sort(attn_tensor,bptt)
#                 sorted_tensor, indices_tensor = torch.sort(attn_tensor,dim=-1,descending=True)
#                 cum_sum = torch.cumsum(sorted_tensor, dim=-1)
#                 bool_mtx = torch.sum(cum_sum>0.9, dim = -1)
#                 count_mtx = bptt+1-bool_mtx
#                 if LOG and args.local_rank == 0:
#                     writer.add_scalar('train_eigen_score'+str(l), tmp_eig, (epoch-1)*num_batches+batch)
#                     writer.add_scalar('train_rank'+str(l), avg_rank, (epoch-1)*num_batches+batch)
#                     writer.add_scalar('train_hopfiled_count'+str(l), count_mtx.float().mean(), (epoch-1)*num_batches+batch)
                    
import numpy as np
def extract_intrinsic(proj_tensor):
    u,s,v=torch.svd(proj_tensor)
    output=(torch.sum(s)/torch.max(s)).cpu().item()
    return output                    

def eigen_sort(attn_tensor_,seq_len_):
    u,s,v=torch.svd(attn_tensor_)
    s=F.normalize(s, p=1, dim=-1)
    sorted_tensor, indices = torch.sort(s, dim=-1,descending=True)
    cum_sum = torch.cumsum(sorted_tensor, dim=-1)
    bool_mtx = torch.sum(cum_sum>0.9, dim = -1)
    count_mtx = sorted_tensor.shape[0]+1-bool_mtx
    
    #     eigen_cnt=[]
#     avg_rank=0
#     size_batch,seq_l,_=attn_tensor_.shape
#     if seq_l==seq_len_:
#         u,s,v=torch.svd(attn_tensor_)
#         sorted_tensor, indices = torch.sort(s, dim=-1,descending=True)

#         for j in range(size_batch):
#             tmp=sorted_tensor[j]
#             tmp=tmp/torch.sum(tmp)
#             avg_rank+=torch.matrix_rank(attn_tensor_[j])
# #             print("yes",avg_rank)
#             for k in range(seq_len_):
#                 if torch.sum(tmp[:k+1])>0.9:
#                     eigen_cnt.append(k)
#                     break
#         try:
            
#             u,s,v=torch.svd(attn_tensor_)
#             sorted_tensor, indices = torch.sort(s, dim=-1,descending=True)
            
#             for j in range(size_batch):
#                 tmp=sorted_tensor[j]
#                 tmp=tmp/torch.sum(tmp)
# #                 avg_rank+=torch.matrix_rank(tmp)
# #                 print("yes",avg_rank)
#                 for k in range(seq_len_):
#                     if torch.sum(tmp[:k+1])>0.9:
#                         eigen_cnt+=k
#                         break
#         except:
#             pass
#         eigen_cnt/=size_batch
#         avg_rank/=size_batch
#     print(eigen_cnt)
    return torch.mean(count_mtx.float()).cpu().item(),0
            
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    avg_rank=0
    eig_score_list=[0 for e in range(nlayers)]
    avg_rank_list=[0 for e in range(nlayers)]
    avg_hopfield_list=[0 for e in range(nlayers)]
    cnt_=0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            for l in range(nlayers):
                attn_tensor=model.transformer_encoder.layers[l].self_attn.multihead_attention.attn_output_weights
                tmp_eig,avg_rank=eigen_sort(attn_tensor,bptt)
                if tmp_eig!=0:  
                    eig_score_list[l]+=tmp_eig
                else:
                    print("here")
                avg_rank_list[l]+=avg_rank

                sorted_tensor, indices_tensor = torch.sort(attn_tensor,dim=-1,descending=True)
                cum_sum = torch.cumsum(sorted_tensor, dim=-1)
                bool_mtx = torch.sum(cum_sum>0.9, dim = -1)
                count_mtx = bptt+1-bool_mtx
                avg_hopfield_list[l]+=count_mtx.float().median()
            cnt_+=1      
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    eig_score_list=[item/cnt_ for item in eig_score_list]
    avg_rank_list=[item/cnt_ for item in avg_rank_list]
    avg_hopfield_list=[item/cnt_ for item in avg_hopfield_list]
#     print(eig_score_list)
    
    
    return total_loss / (len(data_source) - 1),eig_score_list,avg_rank_list,avg_hopfield_list 


# %%


best_val_loss = float("inf")
# epochs = 3 # The number of epochs
best_model = None

val_loss,eig_scores,_,_ = evaluate(model, val_data)
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss,eig_scores,_,_ = evaluate(model, val_data)
    if args.local_rank == 0:
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    if LOG and args.local_rank == 0:
        writer.add_scalar('val_ppl', math.exp(val_loss), epoch)
        for p in range(nlayers):
            writer.add_scalar('val_eigen_score'+str(p), eig_scores[p], epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

#     scheduler.step()


# %%


test_loss,eig_scores,_,_ = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)



