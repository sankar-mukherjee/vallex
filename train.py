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
    parser.add_argument("--metadata_csv_train", type=Path)
    parser.add_argument("--metadata_csv_val", type=Path)
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

def load_datasets(args):
    # train
    train_dataset = TTSDataset(
        args.data_dir, 
        args.metadata_csv_train,
        args.filter_min_duration,
        args.filter_max_duration,
        )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # valid
    val_dataset = TTSDataset(
        args.data_dir, 
        args.metadata_csv_val,
        args.filter_min_duration,
        args.filter_max_duration,
        )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader

def evaluate(model, val_dataloader, loss_criterion, dtype):
    logging.info("Computing validation loss")

    model.eval()

    running_loss = 0.0
    for i, batch in enumerate(val_dataloader):
        audio_embs, audio_lens, text_embs, text_lens, _ = batch
        with torch.cuda.amp.autocast(dtype=dtype):
            _, ar_logits, ar_targets, nar_logits, nar_targets = model(text_embs, text_lens, audio_embs, audio_lens)
            loss, loss_info = loss_criterion(audio_lens, ar_logits, ar_targets, nar_logits, nar_targets)
            assert loss.requires_grad is False
            running_loss += loss
    val_loss = running_loss / len(train_dataloader)

    model.train()
    return val_loss

def train_one_epoch(
        train_dataloader,
        optimizer,
        scheduler,
        scaler,
        args,
        dtype, enabled
    ):
    logging.info("Computing training loss")

    running_loss = 0.0
    progress_bar = tqdm(train_dataloader)
    for i, data in enumerate(progress_bar):
        audio_embs, audio_lens, text_embs, text_lens, speaker = data
        
        args.batch_idx_train += 1
        with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
            with torch.set_grad_enabled(True):
                codes, ar_logits, ar_targets, nar_logits, nar_targets = model(text_embs, text_lens, audio_embs, audio_lens)
                loss, loss_info = loss_criterion(audio_lens, ar_logits, ar_targets, nar_logits, nar_targets)
            assert loss.requires_grad == True

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
        progress_bar.set_description(f"Epoch [{epoch}/{args.num_epochs}]")
        progress_bar.set_postfix(loss=f"{loss:.3f}")
        # print('Batch %d loss: %.3f' % (i + 1, running_loss))

    epoch_loss = running_loss / len(train_dataloader)
    return epoch_loss

if __name__ == "__main__":
    args = parse_argument()
    
    # tensorboard
    tb_writer = SummaryWriter()
    device = get_device()

    # dataset
    train_dataloader, val_dataloader = load_datasets(args)
    # model
    model = VALLE(args.decoder_dim, args.num_heads, args.num_decoder_layers)    
    # loss
    loss_criterion = compute_loss(device)
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

    # training loop
    for epoch in range(args.num_epochs):
        if isinstance(scheduler, Eden): scheduler.step_epoch(epoch - 1)

        train_loss = train_one_epoch(train_dataloader, optimizer, scheduler, scaler, args, dtype, enabled)
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        # print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss))

        # evaluate after each epoch
        val_loss = evaluate(model, val_dataloader, loss_criterion, dtype)
        tb_writer.add_scalar('Loss/valid', val_loss, epoch)

        # save
        save_checkpoint(args.output_dir, epoch, model,optimizer, scheduler, scaler)

    tb_writer.close()
    logging.info("Done!")
    print('inished traning')
