# Adapted from GPT-opt's run_hydra.py: https://github.com/modichirag/GPT-opt (polar branch).
import torch
from defmuon.train_distributed import train
from defmuon.optim.utils import get_scheduler, get_optimizer_factory
from defmuon.utils import set_seed, get_worker_info
from defmuon.model import load_model
from defmuon.dataloader import DATA_DIR, ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import uuid


@hydra.main(version_base=None, config_path="hydra_conf")
def main(config : DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    seed = int(config.get('seed', 42))
    set_seed(seed)
    config['gpt_model']['init_seed'] = seed

    # First set up DDP
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        dist.init_process_group(backend='nccl')
    world_size, rank, local_rank, device = get_worker_info()
    master_process = (rank == 0) # this process will do logging, checkpointing etc.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"Using device: {device}")

    # Logging
    output_dir = HydraConfig.get().runtime.output_dir
    ckpt_dir = f"{output_dir}/checkpoints/" if config['logging_params'].get('save_checkpoint', False) else ""
    if master_process:
        print(f"Loading configuration from {HydraConfig.get().job.config_name}")
        print(f"Training on dataset {config['training_data']['dataset']['name']}")
        os.makedirs(output_dir, exist_ok=True)  
        if ckpt_dir != "": os.makedirs(ckpt_dir, exist_ok=True)

    # Load model
    model = load_model(config['gpt_model'], device)
    # Set the training parameters
    training_params = config['training_data']['training_params']
    opt_config = config["optimizer_params"]
    torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

    # Load data
    dataset_path = DATA_DIR + f"/{config['training_data']['dataset']['name']}-gpt2/"
    if master_process: print(f"Load data from {dataset_path}")
    B, T = training_params['batch_size'], training_params['context_length']
    assert training_params['tokens_processed'] % (world_size * B * T) == 0 
    assert config['logging_params']['val_tokens_processed'] % (world_size * B * T) == 0
    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    # Same cap train() applies, repeated here because the scheduler is built from THIS
    # value: without it the lr schedule would be laid out over the dataset's full length
    # while training stops early, so the run would end mid-decay.
    if training_params.get('max_steps'):
        total_iterations = min(total_iterations, int(training_params['max_steps']))
    if master_process:
        print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
        print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
        print(f"Total number of iterations : {total_iterations}")

    print()
    if master_process:
        print(f"Training with optimizer {opt_config['name']} and learning rate {opt_config['args']['lr']}")

    random_job_id = str(uuid.uuid4()).split('-')[0]
    file_name = f"logs_jobid_{random_job_id}.json"
    output_path = os.path.join(output_dir, file_name)

    # copy model to ensure consistency
    model_copy = copy.deepcopy(model).to(device)
    if training_params['compile']:
        if master_process: print("Compiling model")
        model_copy = torch.compile(model_copy)

    if ddp:
        model_copy = DDP(model_copy, device_ids=[local_rank])

    # Setup optimizer
    optimizer_obj = get_optimizer_factory(opt_config['name'])
    opt_config_args = opt_config['args']
    if opt_config['name'] in ['muon']:
        # Muon consumes (name, param) pairs to route embed/lm_head to its AdamW backup;
        # the torch optimizers (adamw baseline etc.) take the bare parameters.
        optimizer = optimizer_obj(model_copy.named_parameters(), **opt_config_args)
    else:
        optimizer = optimizer_obj(model_copy.parameters(), **opt_config_args)
    scheduler = get_scheduler(config['lr_schedule'], optimizer, total_iterations=total_iterations)

    # Initialize wandb
    if master_process and config['logging_params'].get('wandb', None) is not None:
        config_for_wandb_logging = dict(**config, world_size=world_size)
        wandb_config = config['logging_params']['wandb']
        if "dir" not in wandb_config:
            wandb_config['dir'] = "./outputs/wandb"
        wandb_run = wandb.init(
            **wandb_config,
            id=random_job_id,
            tags=wandb_config.get('tags', []) + [str(HydraConfig.get().job.name)],
            config=config_for_wandb_logging,
            reinit='create_new',
        )
    else:
        wandb_run = None

    # Train
    try:
        logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                    scheduler=scheduler, ckpt_dir=ckpt_dir,
                    logging_params=config['logging_params'], wandb_run=wandb_run)
    finally:
        if master_process and wandb_run is not None:
            wandb_run.finish()

    # Deflation gate diagnostics (only meaningful for the deflated polar methods)
    try:
        from defmuon.optim.deflation_batched import polar_stats
        stats = polar_stats()
        if master_process:
            print(f"polar gate stats: {stats}")
        logger.gate_stats = stats
    except Exception:
        pass

    # Save
    if master_process:
        logger.name = opt_config['name'] + '-lr-' + str(opt_config['args']['lr'])
        if opt_config['name'] == 'muon':
            # `-bd` marks runs whose batched deflation front-end was still active
            # at save time.  A *_cut arm run WITH a cutoff flips it off at the switch
            # step, so its name drops the tag; polar_method already identifies them.
            logger.name = (opt_config['args'].get('polar_method', 'ns5')
                           + '-ns' + str(opt_config['args'].get('ns_steps', 5))
                           + ('-bd' if getattr(optimizer, 'batched_deflation', False) else '')
                           + '-lr-' + str(opt_config['args']['lr']))

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, 'w') as file:
            json.dump(logger.__dict__, file)
        print(f"Saved output to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
