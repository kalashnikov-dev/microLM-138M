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
import time
import torch.nn.functional as F
import torch.nn as nn
import storage
from transformers import get_cosine_schedule_with_warmup

import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False



@dataclass
class SFTConfig:
    filename: str = "model.safetensors"
    repo_id: str = "kalashnikov-dev/microLM-138M"

    epochs: int = 2
    batch_size: int = 2
    accumulation_steps: int = 8
    total_steps: int = 12000 # about 1.5 of the dataset 
    warmup_rate: float = 0.1

    rank: int = 16
    alpha: int = 32
    targets: tuple = ("q", "k", "v", "o", "w1", "w2", "w3") # not list because dataclass doesnt support mutable obj
    
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps_optim: float = 1e-8

    @property
    def warmup_steps(self):
        return int(self.total_steps * self.warmup_rate)
    


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
                chunk_labels = label_buf[:self.seq_len + 1]
                if any(l != -100 for l in chunk_labels): # skip batches with only user tokens 
                    yield (
                        torch.tensor(input_buf[:self.seq_len + 1]),
                        torch.tensor(chunk_labels)
                    )
                input_buf = input_buf[self.seq_len:]
                label_buf = label_buf[self.seq_len:]


def configure_device():
    torch.set_float32_matmul_precision('high')
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


def build_raw_lora_model(device, sft_config, lora_config, model_config):
    base_model = Model(model_config).to(device)
    state_dict = load_file(sft_config.filename)
    base_model.load_state_dict(state_dict)

    raw_lora_model = get_peft_model(base_model, lora_config)
    return raw_lora_model


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

def build_scheduler(optimizer, sft_config):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=sft_config.warmup_steps,
        num_training_steps=sft_config.total_steps
    )
    return scheduler


def build_dataloader(tokenizer, sft_config, model_config, seed):
    stream = load_dataset(
        "HuggingFaceTB/smol-smoltalk", 
        split="train", 
        streaming=True
    ).shuffle(seed=seed, buffer_size=10_000)
    dataset = SFTStreamingDataset(stream, tokenizer, model_config.seq_len)
    dataloader = DataLoader(dataset, batch_size=sft_config.batch_size, pin_memory=True) 

    return dataloader


def build_lora_model(raw_lora_model):
    lora_model = torch.compile(raw_lora_model)
    return lora_model


def train(device, tokenizer, sft_config, model_config, lora_model, raw_lora_model, optimizer, scheduler):
    running_loss = 0.0
    step = 0
    t0 = time.perf_counter()
    for epoch in range(sft_config.epochs):
        dataloader = build_dataloader(tokenizer, sft_config, model_config, 1337+epoch)

        for micro_step, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                pre_logits = lora_model(inputs[:, :-1])
                targets = labels[:, 1:]
                logits = F.linear(pre_logits, raw_lora_model.lm_head.weight)
                loss = F.cross_entropy(logits.reshape(-1, model_config.vocab_size), targets.reshape(-1))
                loss = loss / sft_config.accumulation_steps

            loss.backward()
            running_loss += loss.detach()

            if (micro_step + 1) % sft_config.accumulation_steps == 0:
                grad_norm = nn.utils.clip_grad_norm_(raw_lora_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                #prints
                n_tokens = sft_config.batch_size * model_config.seq_len
                dt = time.perf_counter() - t0
                step_loss = running_loss.item()
                print(f"epoch {epoch} step {step} loss={step_loss:.4f}  grad_norm={grad_norm.item():.2f}  {dt:.2f}s {n_tokens*sft_config.accumulation_steps / dt:.0f} tok/s")
                running_loss = 0.0
                t0 = time.perf_counter()

                if step >= sft_config.total_steps:
                    return




def main():
    device = configure_device()
    tokenizer = build_tokenizer()
    sft_config, lora_config, model_config = build_configs(tokenizer)
    raw_lora_model = build_raw_lora_model(device, sft_config, lora_config, model_config)
    optimizer = build_optimizer(raw_lora_model, sft_config)
    scheduler = build_scheduler(optimizer, sft_config)
    lora_model = build_lora_model(raw_lora_model)

    train(device, tokenizer, sft_config, model_config, lora_model, raw_lora_model, optimizer, scheduler)

    merged = raw_lora_model.merge_and_unload()
    storage.save_final_model(merged, sft_config.repo_id)
    storage.wait_for_uploads()



if __name__ == "__main__":
    main()