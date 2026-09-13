from peft import LoraConfig
from model import Model
from peft import get_peft_model
import torch
from safetensors.torch import load_file
from model import ModelConfig
from transformers import AutoTokenizer
from dataclasses import dataclass

import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False



@dataclass
class SFTConfig:
    filename: str = "checkpoint_82000.safetensors"

    rank: int = 16
    alpha: int = 32
    targets: tuple = ("q", "k", "v", "o", "w1", "w2", "w3") # not list because dataclass doesnt support mutable obj
    
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps_optim: float = 1e-8



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


def build_optimizer(lora_model, lora_config):
    trainable_params = [p for p in lora_model.parameters if p.requires_grad]

    optimizer = torch.optim.AdamW(
    trainable_params,
    lr=lora_config.lr, 
    weight_decay=lora_config.weight_decay,
    betas=lora_config.betas,
    eps=lora_config.eps_optim,
    fused=True
    )
    
    return optimizer


def main():
    device = configure_device()
    tokenizer = build_tokenizer()
    sft_config, lora_config, model_config = build_configs(tokenizer)
    lora_model = build_lora_model(device, sft_config, lora_config, model_config)
    optimizer = build_optimizer(lora_model, lora_config)


    lora_model.print_trainable_parameters()




if __name__ == "__main__":
    main()