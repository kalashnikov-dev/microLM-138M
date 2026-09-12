from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_dataset
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from transformers import AutoTokenizer

import storage
from model import Model, ModelConfig


@dataclass
class TrainConfig:
    batch_size: int = 64
    accumulation_steps: int = 2
    total_steps: int = 105000
    warmup_steps: int = 2000
    lr: float = 2e-3
    repo_id: str = "kalashnikov-dev/miniLM"

    @property
    def total_micro_steps(self):
        return self.total_steps * self.accumulation_steps

    @property
    def decay_steps(self):
        return int(self.total_steps * 0.20)

    @property
    def stable_end(self):
        return self.total_steps - self.decay_steps 



class StreamingDataset(IterableDataset):
    
    def __init__(self, stream, tokenizer, seq_len, batch_size):
        self.stream = stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        buf = []
        batch = []
        for x in self.stream:
            tokens = self.tokenizer.encode(x["text"], add_special_tokens=False)
            buf.extend(tokens)
            buf.append(self.tokenizer.eos_token_id)
                
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf = buf[self.seq_len:]
                batch.append(torch.tensor(chunk))

                if len(batch) == self.batch_size:
                    yield torch.stack(batch)
                    batch = []


class LRMultiplier:
    def __init__(self, warmup_steps, stable_end, decay_steps):
        self.warmup_steps = warmup_steps
        self.stable_end = stable_end
        self.decay_steps = decay_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step < self.stable_end:
            return 1.0
        p = min((step - self.stable_end) / self.decay_steps, 1.0)
        return max(0.0, 1.0 - math.sqrt(p))


def configure_device():
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    return tokenizer


def build_configs(tokenizer):
    train_config = TrainConfig()
    model_config = ModelConfig(vocab_size=len(tokenizer))
    return train_config, model_config


def build_rawmodel(model_config, device):
    raw_model = Model(model_config).to(device)
    return raw_model


def build_optimizer(raw_model, train_config):
    decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': 0.1}, {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=train_config.lr, betas=(0.9, 0.95), eps=1e-8, fused=True
    )
    return optimizer


def build_scheduler(optimizer, train_config):
    scheduler = LambdaLR(optimizer, lr_lambda=LRMultiplier(train_config.warmup_steps, train_config.stable_end, train_config.decay_steps))
    return scheduler


def load_last_checkpoint(raw_model, device, optimizer, scheduler, train_config, model_config):
    resume_micro_step = 0
    docs_to_skip = 0
    ckpt = storage.load_checkpoint(device)
    if ckpt:
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        resume_step = ckpt['step']
        resume_micro_step = resume_step * train_config.accumulation_steps

        tokens_per_step = train_config.batch_size * model_config.seq_len * train_config.accumulation_steps
        tokens_seen = resume_step * tokens_per_step
        AVG_TOKENS_PER_DOC = 1000 
        docs_to_skip = int(tokens_seen / AVG_TOKENS_PER_DOC)

        print(f"loaded checkpoint {resume_step}, skipped {docs_to_skip} documents")

    return resume_micro_step, docs_to_skip


def build_dataloader(docs_to_skip, tokenizer, train_config, model_config):
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-100BT", 
        split="train", 
        streaming=True
    ).skip(docs_to_skip).shuffle(seed=1337, buffer_size=10_000)
    dataset = StreamingDataset(stream, tokenizer, model_config.seq_len, train_config.batch_size)
    dataloader = DataLoader(dataset, batch_size=None, pin_memory=True) 
    return dataloader


def build_model(raw_model):
    model = torch.compile(raw_model)
    return model


def train(raw_model, model, train_config, model_config, device, dataloader, resume_micro_step, optimizer, scheduler, fused_ce):
    running_loss = 0.0
    t0 = time.perf_counter()
    for micro_step, batch in enumerate(dataloader, start=resume_micro_step):
        if micro_step >= train_config.total_micro_steps: 
            break
        
        batch = batch.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            
            pre_logits = model(batch[:, :-1])
            targets = batch[:, 1:]

            loss = fused_ce(raw_model.lm_head.weight, pre_logits.reshape(-1, model_config.d_model), targets.reshape(-1))
            loss = loss / train_config.accumulation_steps

        loss.backward()
        running_loss += loss.detach()

        if (micro_step + 1) % train_config.accumulation_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step = (micro_step + 1) // train_config.accumulation_steps
            if step % 500 == 0:
                storage.save_checkpoint(raw_model, optimizer, scheduler, step, train_config.repo_id)
            
            #prints
            n_tokens = train_config.batch_size * model_config.seq_len
            dt = time.perf_counter() - t0
            step_loss = running_loss.item()
            print(f"step {step} loss={step_loss:.4f}  grad_norm={grad_norm.item():.2f}  {dt:.2f}s {n_tokens*train_config.accumulation_steps / dt:.0f} tok/s")
            running_loss = 0.0
            t0 = time.perf_counter()



def main():
    device = configure_device()
    tokenizer = build_tokenizer()
    train_config, model_config = build_configs(tokenizer)
    raw_model = build_rawmodel(model_config, device)
    optimizer = build_optimizer(raw_model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    fused_ce = LigerFusedLinearCrossEntropyLoss()
    resume_micro_step, docs_to_skip = load_last_checkpoint(raw_model, device, optimizer, scheduler, train_config, model_config)
    dataloader = build_dataloader(docs_to_skip, tokenizer, train_config, model_config)
    model = build_model(raw_model)

    train(
        raw_model, model, train_config, model_config, 
        device, dataloader, resume_micro_step, 
        optimizer, scheduler, fused_ce
    )
    
    storage.save_final_model(raw_model, train_config.repo_id)
    storage.wait_for_uploads()


if __name__ == "__main__":
    main()