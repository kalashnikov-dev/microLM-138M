from peft import LoraConfig
from model import Model
from peft import get_peft_model
import torch
from safetensors.torch import load_file
from model import ModelConfig
from transformers import AutoTokenizer
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False



@dataclass
class SFTConfig:
    filename: str = "checkpoint_82000.safetensors"

    epochs: int = 2
    batch_size: int = 1

    rank: int = 16
    alpha: int = 32
    targets: tuple = ("q", "k", "v", "o", "w1", "w2", "w3") # not list because dataclass doesnt support mutable obj
    
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps_optim: float = 1e-8


#looks complex but basically just masking user queries
class SFTStreamingDataset(IterableDataset):
    def __init__(self, stream, tokenizer, seq_len):
        self.stream = stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    
    def __iter__(self):
        input_buf, label_buf = [], []
        
        for example in self.stream:
            for msg in example["messages"]:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    text = f"User:\n{content}\n\n"
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    input_buf.extend(tokens)
                    label_buf.extend([-100] * len(tokens))
                    
                elif role == "system":
                    text = f"System:\n{content}\n\n"
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    input_buf.extend(tokens)
                    label_buf.extend([-100] * len(tokens))
                    
                elif role == "assistant":
                    text = f"Assistant:\n{content}"
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    
                    tokens.append(self.tokenizer.eos_token_id)
                    
                    sep_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)
                    
                    input_buf.extend(tokens + sep_tokens)
                    label_buf.extend(tokens + [-100] * len(sep_tokens))
                else:
                    continue
            
            while len(input_buf) >= self.seq_len + 1:
                yield (
                    torch.tensor(input_buf[:self.seq_len + 1], dtype=torch.long),
                    torch.tensor(label_buf[:self.seq_len + 1], dtype=torch.long)
                )
                input_buf = input_buf[self.seq_len:]
                label_buf = label_buf[self.seq_len:]


def configure_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    return tokenizer


def build_configs(tokenizer):
    sft_config = SFTConfig() 

    lora_config = LoraConfig(
    r=sft_config.rank,
    lora_alpha=sft_config.alpha,
    target_modules=sft_config.targets,
    bias="none",
    )

    model_config = ModelConfig(vocab_size=len(tokenizer))

    return sft_config, lora_config, model_config


def build_lora_model(device, sft_config, lora_config, model_config):
    base_model = Model(model_config).to(device)
    state_dict = load_file(sft_config.filename)
    base_model.load_state_dict(state_dict)

    lora_model = get_peft_model(base_model, lora_config)
    return lora_model


def build_optimizer(lora_model, sft_config):
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
    trainable_params,
    lr=sft_config.lr, 
    weight_decay=sft_config.weight_decay,
    betas=sft_config.betas,
    eps=sft_config.eps_optim,
    fused=True
    )
    
    return optimizer


def build_dataloader(tokenizer, sft_config, model_config):
    stream = load_dataset(
        "HuggingFaceTB/smol-smoltalk", 
        split="train", 
        streaming=True
    ).shuffle(seed=1337, buffer_size=10_000)
    dataset = SFTStreamingDataset(stream, tokenizer, model_config.seq_len)
    dataloader = DataLoader(dataset, batch_size=sft_config.batch_size, pin_memory=True) 

    return dataloader





def main():
    device = configure_device()
    tokenizer = build_tokenizer()
    sft_config, lora_config, model_config = build_configs(tokenizer)
    lora_model = build_lora_model(device, sft_config, lora_config, model_config)
    optimizer = build_optimizer(lora_model, sft_config)
    dataloader = build_dataloader(tokenizer, sft_config, model_config)


    lora_model.print_trainable_parameters()




if __name__ == "__main__":
    main()