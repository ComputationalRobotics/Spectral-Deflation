# Adapted from GPT-opt: https://github.com/modichirag/GPT-opt (polar branch).
import numpy as np
import torch
import random
import json
import torch.distributed as dist
import os

# get worker info for distributed training
def get_worker_info():
    if dist.is_initialized():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return world_size, rank, local_rank, device
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility

def save_checkpoint(ckpt_dir, step, model, optimizer, loss, dataloader, scheduler=None, keep_last=2):
    world_size = dataloader.world_size
    master_process = (dataloader.rank == 0)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Gather dataloader from all ranks and save it.
    if world_size > 1:
        dataloader_states = [None for _ in range(world_size)]
        state = dataloader.get_state()
        dist.all_gather_object(dataloader_states, state)
    else:
        dataloader_states = [dataloader.get_state()]
    
    # Save the checkpoint and daaloader to a file (e.g. checkpoint.pth)
    if master_process:
        print("Save checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(checkpoint, ckpt_dir + f'/ckpt{step}.pth')
        with open(ckpt_dir + f'/ckpt{step}_loss.json', "w") as f:
            json.dump({'train_loss':loss}, f)

        with open(ckpt_dir + f'/ckpt{step}_dataloader.json', "w") as f:
            json.dump(dataloader_states, f, indent=4)

        # Delete old checkpoints
        manage_checkpoint_directory(ckpt_dir, keep_last)
    

def load_checkpoint(ckpt_dir, step, model, optimizer, dataloader, scheduler=None):
    # Assume the model and optimizer are already created (they should have the same structure)
    print(f"Loading checkpoint {ckpt_dir + f'/ckpt{step}.pth'}")
    checkpoint = torch.load(ckpt_dir + f'/ckpt{step}.pth')

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        if 'scheduler_state_dict' in checkpoint:
            # load_state_dict mutates the scheduler in place and returns None,
            # so it must not be assigned back.
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("WARNING: Scheduler dict not found in checkpoint")

    print(f"Loss at checkpoint : {checkpoint['loss']}")
    with open(ckpt_dir + f'/ckpt{step}_dataloader.json', "r") as f:
        dataloader_states = json.load(f)

    for state in dataloader_states:
        if state['rank'] == dataloader.rank:
            dataloader.set_state(state)

    return model, optimizer, dataloader, scheduler


def manage_checkpoint_directory(ckpt_dir, keep_last=2):
    # List checkpoint files (assuming .pth extension)
    ckpt_files = [os.path.join(ckpt_dir, f)
                        for f in os.listdir(ckpt_dir)
                        if f.endswith(".pth")]
    loss_files = [os.path.join(ckpt_dir, f)
                        for f in os.listdir(ckpt_dir)
                        if f.endswith("_loss.json")]
    
    if not ckpt_files:
        print("No ckpt files found.")
        return
                            
    ckpt_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    recent_ckpts = ckpt_files[:keep_last]

    best_loss = float("inf")
    best_ckpt = None
    
    for file in loss_files:
        try:
            with open(file, "r") as f:
                loss = json.load(f)['train_loss']
            if loss < best_loss:
                best_loss = loss
                best_ckpt = file.split("_loss")[0] + ".pth"
        except Exception as e:
            print(f"Could not load loss from {file}: {e}")

    # Build a set of files to keep: recent ones plus the best-loss one.
    to_keep = set(recent_ckpts)
    if best_ckpt is not None:
        to_keep.add(best_ckpt)
    
    # Delete any ckpt file not in the set to_keep.
    for file in ckpt_files:
        if file not in to_keep:
            os.remove(file)
