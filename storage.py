import torch
import os
import glob
import re
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import save_file

api = HfApi()
executor = ThreadPoolExecutor(max_workers=1)


def _upload(file_path, repo_id):
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type="model"
        )
    except Exception as e:
        print(e)


def save_checkpoint(raw_model, optimizer, scheduler, step, repo_id):
    file_name = f"checkpoint_{step}.pt"
    tmp = file_name + ".tmp"
    
    
    checkpoint = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
    }
    
    torch.save(checkpoint, tmp)
    os.replace(tmp, file_name) # atomic solution

    if step % 5000 == 0:
        executor.submit(_upload, file_name, repo_id)

    #remove old checkpoints
    checkpoints = glob.glob("checkpoint_*.pt")
    checkpoints.sort(key=lambda x: int(re.search(r"checkpoint_(\d+)\.pt", x).group(1)))
    
    for ckpt in checkpoints[:-3]:
        try:
            os.remove(ckpt)
        except OSError:
            pass


def load_checkpoint(device):
    checkpoints = glob.glob("checkpoint_*.pt")
    if not checkpoints:
        return None
        
    latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r"checkpoint_(\d+)\.pt", x).group(1)))
    latest_checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    
    return latest_checkpoint


def save_final_model(raw_model, repo_id):
    file_name = "model.safetensors"
    safe_state_dict = { # safetensors requires contiguous tensors 
        k: v.detach().cpu().contiguous() 
        for k, v in raw_model.state_dict().items()
    }
    save_file(safe_state_dict, file_name)
    executor.submit(_upload, file_name, repo_id)


def wait_for_uploads():
    executor.shutdown(wait=True)