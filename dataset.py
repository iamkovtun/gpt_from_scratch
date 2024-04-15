# MINE
# UPDATE 15/02/7:51:
#fixed logits
#fixed causal mask
#change [translation]
# UPDATE 15/02/8:01:
#fixed dimension in decoder return
#added auto drop
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BillingualDataset(Dataset):

    def __init__ (self, ds, src_lang, tgt_lang, tokenizer_src, tokenizer_tgt, seq_len, auto_drop):
        super().__init__()
        if auto_drop:
            self.ds = [item for item in ds if len(tokenizer_src.encode(item['translation'][src_lang]).ids)+2 <= seq_len and len(tokenizer_tgt.encode(item['translation'][tgt_lang]).ids)+1 <= seq_len]
        else:
            self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len


        self.sos = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        #Extracting pair by index
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]


        #Tokinization sentences and separeting them on src and tgt
        srs_tok_raw = self.tokenizer_src.encode(src_text).ids
        tgt_tok_raw = self.tokenizer_tgt.encode(tgt_text).ids

        #Calculating amount of tokens
        num_src_tok = len(srs_tok_raw)+2
        num_tgt_tok = len(tgt_tok_raw)+1

        num_src_pad = self.seq_len - num_src_tok
        num_tgt_pad = self.seq_len - num_tgt_tok

        #Auto_drop or Checking is amount of tokens not too high

        if num_src_pad < 0 or num_tgt_pad < 0:
            raise ValueError("Sentence too long")


        #Concating tokens
        encoder_input = torch.cat([self.sos,
                            torch.tensor(srs_tok_raw, dtype = torch.int64),
                            self.eos,
                            torch.tensor((num_src_pad*[self.pad]),dtype=torch.int64),
                            ], dim = 0)

        decoder_input = torch.cat([self.sos,
                            torch.tensor(tgt_tok_raw, dtype = torch.int64),
                            torch.tensor((num_tgt_pad*[self.pad]),dtype=torch.int64),
                            ], dim = 0)

        logits = torch.cat([torch.tensor(tgt_tok_raw, dtype = torch.int64),
                            self.eos,
                            torch.tensor((num_tgt_pad*[self.pad]),dtype=torch.int64)
                            ], dim = 0)

        #Double check
        assert self.seq_len == encoder_input.size(0)
        assert self.seq_len == decoder_input.size(0)
        assert self.seq_len == logits.size(0)


        return {
            "src_text":src_text,
            "tgt_text":tgt_text,
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "encoder_mask":(encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask":(decoder_input != self.pad).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), #(1,1,seq_len) & (1,seq_len,seq_len) --> (1,seq_len,seq_len)
            "logits":logits

        }

def casual_mask(size):
    mask = torch.triu(torch.ones((1,size,size)), diagonal=1).type(torch.int)
    return mask == 0
