# Adapted from GPT-opt: https://github.com/modichirag/GPT-opt (polar branch).
# Added here: the compile warm-up before the clock starts, the val-time/step-time
# bookkeeping (val_steps/val_train_times/val_wall_times), and the deflation
# gate/timing hooks.
import torch
import time
import contextlib
import os
import torch.distributed as dist
from defmuon.utils import get_worker_info, save_checkpoint, load_checkpoint
import json

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

class Logging():

    def __init__(self):
        self.losses = []
        self.val_losses = []
        # one entry per val_losses entry, so loss-vs-time plots need no reconstruction:
        # val_steps      : optimizer-step index at which the val loss was measured
        # val_train_times: cumulative sum of step_times up to that point -- pure
        #                  training compute (fwd+bwd+opt), val/logging excluded for
        #                  every arm equally
        # val_wall_times : raw time since training start, everything included
        self.val_steps = []
        self.val_train_times = []
        self.val_wall_times = []
        self.gate_series = []   # cumulative polar_stats() snapshot at every val step
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []
        # per-optimizer-step polar cost, split by phase (DEFMUON_TIME_POLAR=1).
        # deflation_ms = the deflation setup (rSVD + gate + remove-restore); polar_iter_ms = the
        # NS/PE iteration.  Summed over every matrix visited in that step.
        self.deflation_ms = []
        self.polar_iter_ms = []



def eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt):
    # The val loader is intentionally NOT reset between mid-training evals: each
    # eval reads the next val_accum_steps window of the val shard (wrapping at the
    # end).  The schedule is identical for every optimizer arm, so comparisons at
    # equal step index are like-for-like; the end-of-epoch eval resets the loader
    # and averages the full shard (val_accum_steps=0).
    world_size, rank, local_rank, device  = get_worker_info()
    model.eval()
    val_loss = torch.tensor(0., device=device)
    counter = 0
    with torch.no_grad():
        for batch in val_dataloader:
            with autocast_ctxt:
                val_loss += model(batch[0], batch[1], return_logits=False)[1].detach()
            counter += 1
            if (val_accum_steps != 0) and (counter >= val_accum_steps): break
    val_loss = val_loss.detach().clone()/counter
    if world_size > 1: dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(f"Validation Loss: {val_loss.item()}")
    model.train()
    return val_loss


def train(train_dataloader, val_dataloader, model, optimizer, training_params, logging_params, scheduler=None, ckpt_dir="", wandb_run=None):
    
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    # NOTE on units: the loop's `step` counts MICRO-batches.  log_step / val_step /
    # save_ckpt_step are micro-batch counts too, and their branches sit inside the
    # optimizer-step arm, so an interval x fires every lcm(x, grad_accum_steps)
    # micro-batches; the benchmark configs use multiples of grad_accum_steps.
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    # Optional hard cap on optimizer steps.  num_epochs cannot express a fraction (it
    # feeds range()), so this is the only way to run a prefix of a dataset.  It caps
    # total_iterations BEFORE the scheduler is built by the caller's convention, so the
    # lr schedule spans the shortened run rather than the dataset's natural length.
    max_steps = training_params.get('max_steps')
    if max_steps:
        total_iterations = min(total_iterations, int(max_steps))
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')

    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=scheduler)
    if ckpt_dir == "":
        print("Will not save checkpoints as no directory is specified")

    # Compile warm-up: the model's torch.compile and DDP's first-backward bucket
    # setup are one-off costs that would otherwise land inside step 1's step_time,
    # in amounts that vary with the node's inductor-cache state.  One zeros-batch
    # fwd/bwd before the clock pays them here for every arm equally; the eval-mode
    # forward pays the inference graph the first val step would otherwise pay.
    # Zeros token ids are valid inputs, the model has no dropout so no RNG is
    # consumed, no optimizer step runs, and gradients are cleared -- training is
    # unaffected.  DEFMUON_WARM_MODEL=0 disables it.
    if os.environ.get("DEFMUON_WARM_MODEL", "1") == "1":
        try:
            # x and y must be distinct tensors: dynamo guards on the inputs'
            # aliasing pattern, so passing one tensor twice compiles an aliased
            # variant and step 1 recompiles anyway.
            xz = torch.zeros((B, T), dtype=torch.int64, device=device)
            yz = torch.zeros((B, T), dtype=torch.int64, device=device)
            model.train()
            with autocast_ctxt:
                warm_loss = model(xz, yz, return_logits=False)[1]
            warm_loss.backward()
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad(), autocast_ctxt:
                model(xz, yz, return_logits=False)
            model.train()
            del xz, yz, warm_loss
            if master_process:
                print("warm_model: paid model compile before the clock")
        except Exception as e:
            print(f"warm_model: warm-up failed ({e}); step 1 pays the compile "
                  f"as before")

    # Training loop
    train_time_accum = 0.0            # running sum of step_times, for val_train_times
    t_wall_start = time.time()        # for val_wall_times
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        step = 1 if load_ckpt_step == 0 else int(load_ckpt_step)
        optimizer.zero_grad()
        if step != 1: print(train_dataloader.get_state())
        
        for batch in train_dataloader:            
            with autocast_ctxt:
                loss = model(batch[0], batch[1], return_logits=False)[1]
                loss /= grad_accum_steps
            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if step % grad_accum_steps != 0:
                with (model.no_sync() if world_size > 1 else contextlib.nullcontext()):
                    loss.backward()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if world_size > 1: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                    
                #bookkeeping
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                if master_process and wandb_run is not None:
                    wandb_log_dict = {
                        "train/loss": loss_accum.item(), 
                        "train/grad_norm": norm.item(),
                        "train/step_time": step_time,
                        "train/step": step
                    }
                    if hasattr(optimizer, 'step_size_list'):
                        wandb_log_dict["train/step_size_list"] = optimizer.step_size_list
                    for param_group_ix, param_group in enumerate(optimizer.param_groups):
                        wandb_log_dict[f"train/lr_{param_group_ix}"] = param_group['lr']
                    wandb_run.log(wandb_log_dict)
                logger.step_times.append(step_time)
                train_time_accum += step_time
                # the synchronize() above is what makes the CUDA events readable, so
                # draining them here costs nothing extra.  No-op unless timing is armed.
                try:
                    from defmuon.optim.deflation_batched import timing_stats
                    _ts = timing_stats(reset=True)
                    if _ts["calls"]:
                        logger.deflation_ms.append(_ts["clip_ms"])
                        logger.polar_iter_ms.append(_ts["iter_ms"])
                except Exception:
                    pass
                logger.grad_norms.append(norm.item())
                for param_group in optimizer.param_groups:
                    logger.learning_rates.append(param_group['lr'])
                logger.losses.append(loss_accum.item())
                if hasattr(optimizer, 'step_size_list'):  
                    logger.step_size_list = optimizer.step_size_list  
                
                if (step % logging_params['log_step'] == 0) & master_process:
                    tps = training_params["tokens_processed"] / step_time
                    print(f"Step {step} of {total_iterations*grad_accum_steps}.")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f}k | Loss : {loss_accum.item():0.3f}")
                    
                if (step % logging_params['val_step'] == 0):
                    val_loss = eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt)
                    if master_process and wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss.item(), "val/step": step})
                    logger.val_losses.append(val_loss.item())
                    logger.val_steps.append(step // grad_accum_steps)
                    logger.val_train_times.append(train_time_accum)
                    logger.val_wall_times.append(time.time() - t_wall_start)
                    try:
                        from defmuon.optim.deflation_batched import polar_stats
                        logger.gate_series.append(polar_stats())
                    except Exception:
                        pass

                if (step % logging_params['save_ckpt_step'] == 0) & (ckpt_dir != ""):
                    save_checkpoint(ckpt_dir, step, model, optimizer, loss_accum.item(),
                                    train_dataloader, scheduler, logging_params['keep_last'])
                    
                    if master_process:
                        with open(ckpt_dir + '/log.json', 'w') as file:
                            json.dump(logger.__dict__, file)
                loss_accum = 0.
                start_time = time.time() 
            if step // grad_accum_steps >= total_iterations:
                break
            step += 1
            
            
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set, and save final values
        val_dataloader.reset()
        val_loss = eval_validation_loss(model, val_dataloader, 0, autocast_ctxt)
        logger.val_losses.append(val_loss.item())
        logger.val_steps.append(step // grad_accum_steps)
        logger.val_train_times.append(train_time_accum)
        logger.val_wall_times.append(time.time() - t_wall_start)
        try:
            from defmuon.optim.deflation_batched import polar_stats
            logger.gate_series.append(polar_stats())
        except Exception:
            pass
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}")        
        if (ckpt_dir != ""):
            save_checkpoint(ckpt_dir, step, model, optimizer, logger.losses[-1],
                        train_dataloader, scheduler, logging_params['keep_last'])        
            if master_process:
                with open(ckpt_dir + '/log.json', 'w') as file:
                    json.dump(logger.__dict__, file)
        if master_process and wandb_run is not None:
            wandb_run.log({"val/loss": val_loss.item(), "val/step": step, "train/loss": logger.losses[-1], "train/step": step})

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    return logger
