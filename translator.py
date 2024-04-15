import torch
from config import get_latest_weight, get_weights_file_path, get_config
from train import get_or_build_tokinizer, greedy_decode
from model import build_transformer

def set_translator():
    return {"epoch": "latest",
            "src_lang":"en",
            "target_lang":"it"}

def translate(text:str):
    config = get_config()
    settings = set_translator()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
 
    tokenizer_src = get_or_build_tokinizer(config, settings["src_lang"])
    tokenizer_tgt = get_or_build_tokinizer(config, settings["target_lang"])

    model = build_transformer(tokenizer_src.get_vocab_size(),
                            tokenizer_tgt.get_vocab_size(),
                            config["seq_len"],
                            config["seq_len"],
                            config["d_model"],
                            config["head_numbers"],
                            config["dropout"],
                            config["N"],
                            config["dff"]).to(device)


    model_filename = get_latest_weight(config) if settings["epoch"] == "latest" else get_weights_file_path(config, config['preload'])
    if model_filename is not None:
        print(f'Loading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("Weights are not found")
    model.eval()
    
    with torch.no_grad():
        text_tokens = tokenizer_src.encode(text).ids
        encoder_input = torch.cat([
                    torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                    torch.tensor(text_tokens, dtype = torch.int64),
                    torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                    torch.tensor(((config["seq_len"]-len(text_tokens)-2)*[torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)]),dtype=torch.int64),
                    ], dim = 0).unsqueeze(0)

        encoder_mask = (encoder_input != torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)).unsqueeze(0).unsqueeze(0).int().to(device)

        encoder_input = encoder_input.to(device)

        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        
        model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config["seq_len"], device)
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        return model_out_text

out=translate("He bought it")
print(out)







# define model
# load weights and model
# compute encoder input once
# compute decoder input and each itteration until reaced seq_len or eos token
