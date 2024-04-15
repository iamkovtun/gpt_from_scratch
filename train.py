# MINE
# UPDATE 16/02:01:59:
# get sentence function
# zero.grad
# UPDATE:
# added deleating previous files
# added auto_drop

from config import get_config,  get_weights_file_path, get_latest_weight
from model import build_transformer
from dataset import BillingualDataset, casual_mask


import matplotlib.pyplot as plt
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from torch.utils.tensorboard import SummaryWriter
import torchmetrics


from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer


from tqdm import tqdm

from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    #compute the encoder output which will be used in each itteration of decoder
    encoder_output = model.encode(source, source_mask)
    #initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        #Build mask for the target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #Calculate the output of the decoder
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        #Get the next token
        probs = model.project(decoder_output[:,-1])

        #Selecet the token with the max probability(greedy search)
        _, next_word = torch.max(probs, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)




def validation_loop(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples = 2):
    counter = 0
    console_width = 80
    model.eval()

    source = []
    expected = []
    predicted = []


    with torch.no_grad():
        for batch in val_dataloader:
            counter += 1
            encoder_mask = batch["encoder_mask"].to(device)
            encoder_input = batch["encoder_input"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            if writer:
                # Evaluate the character error rate
                # Compute the char error rate 
                metric = torchmetrics.CharErrorRate()
                cer = metric(predicted, expected)
                writer.add_scalar('validation cer', cer, global_step)
                writer.flush()

                # Compute the word error rate
                metric = torchmetrics.WordErrorRate()
                wer = metric(predicted, expected)
                writer.add_scalar('validation wer', wer, global_step)
                writer.flush()

                # Compute the BLEU metric
                metric = torchmetrics.BLEUScore()
                bleu = metric(predicted, expected)
                writer.add_scalar('validation BLEU', bleu, global_step)
                writer.flush()


            #Print massage
            print_msg("-"*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if counter == num_examples:
                break



def get_ds(config):
    #download dataset
    ds_raw = load_dataset(f"{config['dataset_name']}", f"{config['lang_src']}-{config['lang_tgt']}", split = "train")

    #split dataset(90% to train, 10% to val)
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    #create tokinizers
    tokenizer_src = get_or_build_tokinizer(config, config["lang_src"], ds_raw)
    tokenizer_tgt= get_or_build_tokinizer(config, config["lang_tgt"], ds_raw)

    #cheacking dataset tokinized inputs to measure neccecary seq_len if need
    if config["seq_len"] == None:
        all_src_tok = []
        all_tgt_tok = []
        for item in ds_raw:
            src_tok = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
            tgt_tok = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids

            all_src_tok.append(len(src_tok))
            all_tgt_tok.append(len(tgt_tok))
        
        # Plotting the distribution of token lengths
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(all_src_tok, bins=30, color='blue', alpha=0.7)
        plt.title('Source Token Length Distribution')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(all_tgt_tok, bins=30, color='green', alpha=0.7)
        plt.title('Target Token Length Distribution')
        plt.xlabel('Token Length')

        plt.tight_layout()
        plt.show()

        
        seq_len = int(input("New Sequense Lenght: "))

        srs_procent_droped = len([x for x in all_src_tok if x > seq_len - 2]) / len(all_src_tok) * 100
        tgt_procent_droped = len([x for x in all_tgt_tok if x > seq_len - 1]) / len(all_src_tok) * 100

        #seq_len = max(max_src_tok,max_tgt_tok) + 2
        #while seq_len % 10 != 0:
        #    seq_len += 1

        config["seq_len"] = seq_len

        print(f"Procent of source tokens dropped {srs_procent_droped:.3f}")
        print(f"Procent of target tokens dropped {tgt_procent_droped:.3f}")
        print("Seq_Len defined as", seq_len)
        


    #create tokinized datasets
    train_ds = BillingualDataset(train_ds_raw, config["lang_src"], config["lang_tgt"], tokenizer_src, tokenizer_tgt, config["seq_len"], config["auto_drop"])
    val_ds = BillingualDataset(val_ds_raw, config["lang_src"], config["lang_tgt"], tokenizer_src, tokenizer_tgt, config["seq_len"], config["auto_drop"])



    #create inputs
    train_dataloader = DataLoader(train_ds,batch_size = config["batch_size"], shuffle= True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle= True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_sentences(lang, ds):
    for item in ds["translation"]:
        yield item[lang]


def get_or_build_tokinizer(config, lang, ds = None):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency = 2)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(get_sentences(lang, ds), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer


def train_model(config):
        # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    device = torch.device(device)

    #cheaking path
    Path(f'{config["dataset_name"]}_{config["model_folder"]}').mkdir(parents = True, exist_ok = True)


    #creating ds and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    #creating the model, optimizer and loss
    model = build_transformer(tokenizer_src.get_vocab_size(),
                              tokenizer_tgt.get_vocab_size(),
                              config["seq_len"],
                              config["seq_len"],
                              config["d_model"],
                              config["head_numbers"],
                              config["dropout"],
                              config["N"],
                              config["dff"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'],eps=1e-9)
    loss_fn = nn.CrossEntropyLoss( ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    initial_epoch = 0
    global_step = 0

    #Tensorboard
    writer = SummaryWriter(f"runs/{config['experiment_name']}")


    #resume training in case of crash

    if config["preload"]:
        model_filename = get_latest_weight(config) if config["preload"] == "latest" else get_weights_file_path(config, config['preload'])
        if model_filename is not None:
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            initial_epoch = state['epoch'] + 1
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print("Train from scratch...")
        



    #training loop
    for epoch in range(initial_epoch, config["num_of_epoches"]):
        torch.cuda.empty_cache()
        #set model to train. it behaves differnt in train and val modes
        model.train()
        #wrapping DataLoader object for process visualisation
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            #extacting input data from from DataLoader
            encoder_input = batch["encoder_input"].to(device) # (B, Seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B, Seq_len)
            encoder_mask  = batch["encoder_mask"].to(device) # (B, 1, 1, Seq_len)
            decoder_mask  = batch["decoder_mask"].to(device) # (B, 1, Seq_len, Seq_len)

            #Run data tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) #(B, Seq_Len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) #(B, Seq_Len, d_model)
            projected_output = model.project(decoder_output) # (B, Seq_Len, tgt_vocab_size)

            logits = batch["logits"].to(device) #(B, Seq_len)

            #(B, Seq_len, tgt_vocab_size) --> (B * Seq_len, tgt_vocab_size)
            loss = loss_fn(projected_output.view(-1, tokenizer_tgt.get_vocab_size()), logits.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            #Addidng loss in TensorBoard
            writer.add_scalar("loss", loss, global_step)
            writer.flush()

            #Backpropagate the loss
            loss.backward()

            #Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1          

        validation_loop(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples = 25)


        #Save the model each 1 step
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)


if __name__ == '__main__':
    if 'config' not in globals():
        config = get_config()
    train_model(config)





