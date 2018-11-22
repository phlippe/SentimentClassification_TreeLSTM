import re
import random
import time
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
plt.style.use('default')


class AttentionModule(nn.Module):
    """Realizes a very simple attention module. Takes last state and word into account with two-layer feedforward"""
    
    def __init__(self, embedding_dim, state_dim, hidden_dim):
        super(AttentionModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.output_layer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(embedding_dim+state_dim, hidden_dim)),
            ('tanh', nn.Tanh()),
            ('dropout', nn.Dropout(0.25)),
            ('fc2', nn.Linear(hidden_dim, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, word_embed, last_state):
        
        input_tensor = torch.cat([word_embed, last_state], dim=1)
        
        for layer in self.output_layer:
            input_tensor = layer(input_tensor)
        
        return input_tensor
        

class IterativeLSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab, iterations):
        super(IterativeLSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)
        self.attention_module = AttentionModule(embedding_dim, hidden_dim, embedding_dim)
        self.iterations = iterations

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # time (the number of words in the sentence)

        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)

        # process input sentences one word/timestep at a time
        # input is batch-major, so the first word(s) is/are input_[:, 0]
        for iter_ind in range(self.iterations):
            if iter_ind > 0:
                # hx has dimension B,hidden_size -> need to be B,T,hidden_size
                exp_hx = hx.unsqueeze(dim=1).repeat([1, T, 1])
                attent_hx_input = exp_hx.view((B * T, self.rnn.hidden_size))
                attent_word_input = input_.view((B * T, self.embedding_size))
                print(attent_hx_input)
                print(attent_word_input)
                attent_scores = self.attention_module(attent_word_input, attent_hx_input)
                attent_scores = attent_scores.reshape(B, T, 1)

            outputs = []
            for i in range(T):
                hx_new, cx_new = self.rnn(input_[:, i], (hx, cx))
                if iter_ind == 0:
                    hx = hx_new
                    cx = cx_new
                else:
                    hx = attention_score[:, i] * hx_new + (1 - attention_score[:, i]) * hx
                    cx = attention_score[:, i] * cx_new + (1 - attention_score[:, i]) * cx
                outputs.append(hx)

            # if we have a single example, our final LSTM state is the last hx
            if B == 1:
                last_final_states = hx
            else:
                #
                # This part is explained in next section, ignore this else-block for now.
                #
                # we processed sentences with different lengths, so some of the sentences
                # had already finished and we have been adding padding inputs to hx
                # we select the final state based on the length of each sentence

                # two lines below not needed if using LSTM form pytorch
                outputs = torch.stack(outputs, dim=0)  # [T, B, D]
                outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

                # to be super-sure we're not accidentally indexing the wrong state
                # we zero out positions that are invalid
                pad_positions = (x == 1).unsqueeze(-1)

                outputs = outputs.contiguous()
                outputs = outputs.masked_fill_(pad_positions, 0.)

                mask = (x != 1)  # true for valid positions [B, T]
                lengths = mask.sum(dim=1)  # [B, 1]

                indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
                last_final_states = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

            hx = last_final_states
            cx = input_.new_zeros(B, self.rnn.hidden_size)

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(last_final_states)
        return logits

torch.cuda.empty_cache()
lstm_model = IterativeLSTMClassifier(
    len(v.w2i), 300, 168, len(t2i), v, 2)

# copy pre-trained vectors into embeddings table
with torch.no_grad():
    lstm_model.embed.weight.data.copy_(torch.from_numpy(vectors))
    lstm_model.embed.weight.requires_grad = False

print(lstm_model)
print_parameters(lstm_model)  
  
lstm_model = lstm_model.to(device)

batch_size = 32
optimizer = optim.Adam(lstm_model.parameters(), lr=2e-4)

lstm_losses, lstm_accuracies = train_model(
    lstm_model, optimizer, num_iterations=30000, 
    print_every=250, eval_every=250,
    batch_size=batch_size,
    batch_fn=get_minibatch, 
    prep_fn=prepare_minibatch,
    eval_fn=evaluate)