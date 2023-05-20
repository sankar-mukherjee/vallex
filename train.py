import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vallex.dataset import TTSDataset, collate_fn
from vallex.loss import compute_loss
# from vallex.model import VALLE
from vallex.model2 import VALLE
from vallex.optim import Eden
from vallex.utils import (
    load_checkpoint, save_checkpoint, 
    get_optimizer, get_device, get_autocast_type
)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--metadata_csv", type=Path)
    parser.add_argument("--output_dir", type=Path)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--decoder_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_decoder_layers", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--warmup_steps", type=int, default=200)   
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dtype", type=str, default='float16')
    parser.add_argument("--accumulate-grad-steps", type=int, default=1, help="""update gradient when batch_idx_train % accumulate_grad_steps == 0.""")
    parser.add_argument("--batch_idx_train", type=int, default=0)

    parser.add_argument("--filter_min_duration", type=float, default=0.0)
    parser.add_argument("--filter_max_duration", type=float, default=100.0)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_argument()
    
    device = get_device()

    dataset = TTSDataset(
        args.data_dir, 
        args.metadata_csv,
        args.filter_min_duration,
        args.filter_max_duration,
        )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # model
    model = VALLE(args.decoder_dim, args.num_heads, args.num_decoder_layers)    
    # loss
    criterion = compute_loss(device)
    # optimizer
    optimizer = get_optimizer(model, args.learning_rate)
    # scheduler
    scheduler = Eden(optimizer, 5000, 4, warmup_batches=args.warmup_steps)
    # scaler
    scaler = GradScaler(enabled=(args.dtype in ["fp16", "float16"]), init_scale=1.0)
    
    # check previous chekpoints
    checkpoints = load_checkpoint(
        args.output_dir, 
        model,
        optimizer,
        scheduler,
        scaler,
        )
    model.to(device)

    # autocast_type
    dtype, enabled = get_autocast_type(args.dtype)

    writer = SummaryWriter()
    is_training = True
    num_epochs = args.num_epochs
    # training loop
    for epoch in range(num_epochs):
        if isinstance(scheduler, Eden):
            scheduler.step_epoch(epoch - 1)

        running_loss = 0.0
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            audio_embs, audio_lens, text_embs, text_lens, speaker = data
            
            args.batch_idx_train += 1
            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                with torch.set_grad_enabled(is_training):
                    codes, ar_logits, ar_targets, nar_logits, nar_targets = model(text_embs, text_lens, audio_embs, audio_lens)
                    loss, loss_info = criterion(audio_lens, ar_logits, ar_targets, nar_logits, nar_targets)
                assert loss.requires_grad == is_training

            if args.batch_idx_train >= args.accumulate_grad_steps:
                if args.batch_idx_train % args.accumulate_grad_steps == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    for k in range(args.accumulate_grad_steps):
                        if isinstance(scheduler, Eden):
                            scheduler.step_batch(args.batch_idx_train)
                        else:
                            scheduler.step()

            running_loss += loss
            
            # add stuff to progress bar in the end
            progress_bar.set_description(f"Epoch [{epoch}/{num_epochs}]")
            progress_bar.set_postfix(loss=f"{loss:.3f}")
            # print('Batch %d loss: %.3f' % (i + 1, running_loss))

        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss))

        save_checkpoint(args.output_dir, epoch, model,optimizer, scheduler, scaler)

    writer.close()
