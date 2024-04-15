import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        #Create Embedding layer. Each token in vocabulary will have d_model(512) vector which are trained parameters.Each number in vector represent some feature of token. The structur helps to understand meaning of tokens and their similarity.
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        #As in paper multiply embedding layer by sqrt of d_model
        out = self.embedding(x) * math.sqrt(self.d_model)
        self.out = out
        return self.out

class PosEmbedding(nn.Module):

    def __init__(self, seq_len:int, d_model:int, dropout:float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        #Creating matrix (seq_len,d_model).Similary to tokens embedding, each position will represents to d_model vector. But for now all zeros
        pe = torch.zeros(seq_len, d_model)
        #Creating positions vector and makaing it matrix (seq_len,1)
        pos = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1)
        #Create denominator for function in paper but upgraded with log for numerical stability. !Not sure how does this function substitut the that in paper.
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        #Make positional encodind similar to paper
        pe[:,0::2] = torch.sin(pos*denominator)
        pe[:,1::2] = torch.cos(pos*denominator)
        #Add extra dim for batches. Not sure why we add it here.
        pe = pe.unsqueeze(0)
        #Make pe untrainable parameters similar to running mean in BatchNorm. With this method also save parameters in model output parameters file. It also saves pe sa self.pe
        self.register_buffer("pe", pe)

    def forward(self,x):
        #Cobine inputembedding with pe and adjust no grad.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Tutorial:(x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False))in tutorial he makes slice of pe but slice same with raw pe.  Make in my own way, seems right
        self.out = self.dropout(x)
        return self.out

class LayerNorm(nn.Module):

    def __init__(self, d_model:int, eps:float=10**-6):
        super().__init__()
        #to protect from gradient exploding and errors cause deviding by zero
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))#, device=config["device"] ))
        self.beta = nn.Parameter(torch.zeros(d_model))#, device=config["device"]))

    def forward(self,x):
        #x is (batches, seq_len, hidden_layer_size)
        mean = x.mean(dim = -1,keepdim = True)# mean of each last dim row and keepdim for further multiplication
        std = x.std(dim = -1,keepdim = True)#same but with std
        #normalization as in function
        self.out = self.alpha * (x-mean) / (std + self.eps) + self.beta
        return self.out

class FeedForward(nn.Module):

    def __init__(self,dff:int, d_model:int, dropout: float):
        super().__init__()
        #dff is dim of inner liner function in feed forward
        self.linear1 = nn.Linear(d_model,dff)
        self.linear2 = nn.Linear(dff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        #similar to paper x(batches, seq_len, d_model)--> (batches, seq_len, dff) --> out(batches, seq_len, d_model)
        x = self.linear1(x)
        x = self.dropout(torch.relu(x))
        self.out = self.linear2(x)
        return self.out

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        #checking possibility of provided hyperparameters
        assert d_model % h == 0, "h and d_model are not devisiable"
        #creating size heads, by separating embeddings
        self.d_k = d_model // h
        self.h = h

        #creating parameters for queries, keys, values
        self.w_q = nn.Linear(d_model,d_model, bias = False)
        self.w_k = nn.Linear(d_model,d_model, bias = False)
        self.w_v = nn.Linear(d_model,d_model, bias = False)
        #creatings parameters for output
        self.w_o = nn.Linear(d_model,d_model, bias = False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(queries, keys, values, mask, dropout: nn.Dropout):
        d_k = queries.shape[-1]

        attention_raw = (queries @ keys.transpose(-2, -1)) / (math.sqrt(d_k)) #(batch, h, seq_len, d_k) @ (batch, h, seq_len, d_k).transpose(-2, -1) --> (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_raw.masked_fill_(mask == 0, -float('inf'))# if object in mask == 0 fill same object in attention_raw with value -inf

        attention_score = attention_raw.softmax(dim = -1)#getting scores of how tokens are relate to each other(last dim)
        if dropout is not None:
            attention_raw =  dropout(attention_raw)

        # as in formula
        return attention_score, (attention_score @ values) #(batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)

    def forward(self, q, k, v, mask):
        #appling Linears
        queries = self.w_q(q)
        keys = self.w_k(k)
        values = self.w_v(v)

        # splitting embeddings on heads, and transpose in way that each head can see all sentence but only particular part of embedding
        queries = queries.view(queries.shape[0], queries.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_len, d_model) --> (batch, seq_len, h, dK) --> (batch, h, seq_len, d_k)
        keys = keys.view(keys.shape[0], keys.shape[1], self.h, self.d_k).transpose(1,2) # same as above
        values = values.view(values.shape[0], values.shape[1], self.h, self.d_k).transpose(1,2) # same as above

        #calculating attention using formula
        attention_score, x  = MultiHeadAttention.attention(queries, keys, values, mask, self.dropout)
        # as concuttinate all heads(combinning)
        x = x.transpose(1,2).contiguous()# to concuttinate correctly. (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k). contiguous needs to save x in mamory so view can be aplyied correctly
        x = x.view(x.shape[0],x.shape[1],self.h*self.d_k) # (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        self.out = self.w_o(x)
        return self.out

class RisidualConnection(nn.Module):

    def __init__(self,d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        #I dont know why he put num of batches here when it should be d_model
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        normalized = self.norm(x) #normalization
        sublayered = sublayer(normalized) #makeing output from main path
        droped = self.dropout(sublayered)
        self.out = x + droped #adding
        return self.out


class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feedforward: FeedForward, d_model:int, dropout:float):#creating one EncoderBlock.
        super().__init__()
        # define all necessary modules that later will be used in forward method
        self.feedforward = feedforward # is created in Transformer class and passed here
        self.self_attention = self_attention# is created in Transformer class and passed here
        self.risidualconnection = nn.ModuleList([RisidualConnection(d_model, dropout) for _ in range(2)])# created here

    def forward(self, x, src_mask):
        x = self.risidualconnection[0](x, lambda x: self.self_attention(x,x,x,src_mask))# resend created attention to risidual connection where rd will be called. Made kind of transfer. We receive it send in further to rd. But we use lambda because of construction of RD so it wil able to call it.
        self.out = self.risidualconnection[1](x, self.feedforward) #resend created attention to risidual connection where rd will be called. Same as above but without lambda
        # we send all necessities to RD. It calculate it and we received result from RD
        return self.out

class Encoder(nn.Module):
    # run all layers(Its basicaly sequance of N amount encoder blocks as defined in Transformer. And apply norm
    def __init__(self, d_model, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self,x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.norm(x)
        self.out = x
        return self.out

class DecoderBlock(nn.Module):

    def __init__(self, self_attention:MultiHeadAttention, cross_attention:MultiHeadAttention, feedforward:FeedForward, d_model:int, dropout:float):
        super().__init__()
        #same as in encoder block but plus one MultiHeadAttentin which is cross attention.
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.risidual_connection = nn.ModuleList([RisidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self,x,encod_out,src_mask, tgt_mask):
        x = self.risidual_connection[0](x,lambda x : self.self_attention(x,x,x,tgt_mask))
        x = self.risidual_connection[1](x,lambda x : self.cross_attention(x,encod_out,encod_out,src_mask))# here we send quiries of self_attention drom decoder and keys and values from encoder self attention. Not sure why values and keys are the same(encoder_output)
        x = self.risidual_connection[2](x, self.feedforward)
        self.out = x
        return self.out

class Decoder(nn.Module):
    # run all layers(Its basicaly sequance of N amount decoder blocks as defined in Transformer. And apply norm
    def __init__(self, d_model, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self,x, encod_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encod_out, src_mask, tgt_mask)

        x = self.norm(x)
        self.out = x
        return self.out

class ProjectionLayer(nn.Module):
    # as in paper last activation functions(linear)
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        self.out = x
        return self.out


class Transformer(nn.Module):
    #combine Encoder, Decoder and Projections together. Make it separetely because eccoder output we need to save and do not compute many times
    def __init__(self, src_embedding:InputEmbedding,  src_pos:PosEmbedding, encoder:Encoder, tgt_embedding:InputEmbedding, tgt_pos:PosEmbedding, decoder:Decoder, projection:ProjectionLayer):
        super().__init__()
        self.src_embedding = src_embedding
        self.src_pos = src_pos
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.tgt_pos = tgt_pos
        self.decoder = decoder
        self.projection = projection

    def encode(self,x, src_mask):
        src = self.src_embedding(x)
        src = self.src_pos(src)
        encoder_ouput = self.encoder(src, src_mask)
        return encoder_ouput

    def decode(self,x, encod_out, src_mask, tgt_mask):
        tgt = self.tgt_embedding(x)
        tgt = self.tgt_pos(tgt)
        decoder_output = self.decoder(tgt, encod_out, src_mask, tgt_mask)
        return decoder_output

    def project(self, x):
        return self.projection(x)

def build_transformer(src_voc_size, tgt_voc_size, src_seq_len, tgt_seq_len, d_model, h, dropout:float, N:int, dff:int):
    #Create embeddings
    src_embedding = InputEmbedding(d_model, src_voc_size)
    tgt_embedding = InputEmbedding(d_model, tgt_voc_size)

    #Create pos embedding
    src_pos = PosEmbedding(src_seq_len,d_model,dropout)
    tgt_pos = PosEmbedding(tgt_seq_len,d_model,dropout)

    #Create encodding blocks
    encoding_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        feedforward = FeedForward(dff,d_model,dropout)
        encoding_block = EncoderBlock(self_attention,feedforward,d_model,dropout)
        encoding_blocks.append(encoding_block)

    #Create decodding blocks
    decoding_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        feedforward = FeedForward(dff,d_model,dropout)
        decoding_block = DecoderBlock(self_attention,cross_attention,feedforward, d_model, dropout)
        decoding_blocks.append(decoding_block)

    #Create Encoder and Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoding_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoding_blocks))

    #Create Projection
    projection = ProjectionLayer(d_model, tgt_voc_size)

    #Create Transformer
    transformer = Transformer(src_embedding, src_pos, encoder, tgt_embedding, tgt_pos, decoder, projection)

    #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
