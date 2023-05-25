import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

import utils.logger as logger
from embeddings.collation import get_text_token_collater
from embeddings.tokenizer import (AudioTokenizer, TextTokenizer,
                                  tokenize_audio, tokenize_text)
from vallex.dataset import TTSDataset, collate_fn
from vallex.loss import compute_loss
from vallex.model import VALLE
from vallex.utils import (get_autocast_type, get_device, get_mel_specgram,
                          get_optimizer, get_scaler, get_scheduler,
                          load_checkpoint, read_audio_waveform,
                          save_checkpoint)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--metadata_csv_train", type=Path)
    parser.add_argument("--metadata_csv_val", type=Path)
    parser.add_argument("--unique_text_tokens", type=Path)    
    parser.add_argument("--output_dir", type=Path)

    parser.add_argument("--scheduler_type", type=str, default='Eden')
    parser.add_argument("--optimizer_type", type=str, default='ScaledAdam')
    parser.add_argument("--scaler_type", type=str, default='GradScaler')
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--decoder_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_decoder_layers", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--warmup_steps", type=int, default=200)   
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dtype", type=str, default='float16')
    parser.add_argument("--grad_accumulation", type=int, default=1)
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
        args.unique_text_tokens,
        args.filter_min_duration,
        args.filter_max_duration,
        )
    # valid
    val_dataset = TTSDataset(
        args.data_dir, 
        args.metadata_csv_val,
        args.unique_text_tokens,
        args.filter_min_duration,
        args.filter_max_duration,
        )
    
    if args.debug:
        # debug
        print('Running in debugg mode')
        subset_indices = list(range(100))
        sampler = SubsetRandomSampler(subset_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader

def synthesize(model, audio_prompt, text_prompt, text, unique_text_tokens, device):
    model.eval()
    audio_tokenizer = AudioTokenizer()
    text_tokenizer = TextTokenizer()
    text_collater = get_text_token_collater(unique_text_tokens)

    # synthesis
    # tokenize text
    tokenize_text_ = tokenize_text(text_tokenizer, text=f"{text_prompt} {text}".strip())
    text_tokens, text_tokens_lens = text_collater([tokenize_text_])
    _, enroll_x_lens = text_collater([tokenize_text(text_tokenizer, text=f"{text_prompt}".strip())])
    # tokenize audio
    audio_emb = tokenize_audio(audio_tokenizer, audio_prompt)
    audio_emb = audio_emb[0][0].transpose(2, 1)

    # infer
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_emb.to(device),
        enroll_x_lens=enroll_x_lens.to(device),
    )
    samples = audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
    samples = samples[0,:]
    return samples

def evaluate(model, val_dataloader, loss_criterion, total_iter, dtype):
    model.eval()

    print('Running validation...')
    running_loss = 0.0
    for batch in tqdm(val_dataloader):
        audio_embs, audio_lens, text_embs, text_lens, _ = batch
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.set_grad_enabled(False):
                _, ar_logits, ar_targets, nar_logits, nar_targets, nar_loss_norm_factor, total_length = model(text_embs, text_lens, audio_embs, audio_lens)
                loss, loss_info = loss_criterion(ar_logits, ar_targets, nar_logits, nar_targets, nar_loss_norm_factor, total_length)
                assert loss.requires_grad is False
        running_loss += loss/(audio_lens).sum()

    val_loss = running_loss / len(val_dataloader)
    # logs
    logger.log(total_iter, subset='val', data=OrderedDict([('loss', val_loss)]))
    return val_loss

def train_one_epoch(
        train_dataloader,
        model,
        optimizer,
        scheduler,
        scaler,
        args,
        epoch, total_iter,
        dtype, enabled
    ):

    running_loss = 0.0
    accumulated_steps = 0

    progress_bar = tqdm(train_dataloader)
    for i, data in enumerate(progress_bar):
        audio_embs, audio_lens, text_embs, text_lens, speaker = data
        
        args.batch_idx_train += 1
        with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
            with torch.set_grad_enabled(True):
                codes, ar_logits, ar_targets, nar_logits, nar_targets, nar_loss_norm_factor, total_length = model(text_embs, text_lens, audio_embs, audio_lens)
                loss, loss_info = loss_criterion(ar_logits, ar_targets, nar_logits, nar_targets, nar_loss_norm_factor, total_length)
            assert loss.requires_grad == True

        scaler.scale(loss).backward()
        accumulated_steps += 1
        if accumulated_steps % args.grad_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            for k in range(args.grad_accumulation):
                if scheduler.__class__.__name__ == 'Eden':
                    scheduler.step_batch(args.batch_idx_train)
                else:
                    scheduler.step()

            total_iter += 1
            accumulated_steps = 0
            # logs
            logger.log(total_iter, subset='train', data=OrderedDict([('loss', loss/(audio_lens).sum())]))
            logger.log(total_iter, subset='train', data=OrderedDict([('lrate', optimizer.param_groups[0]['lr'])]))

        running_loss += loss/(audio_lens).sum()
        
        # add stuff to progress bar in the end
        progress_bar.set_description(f"Epoch [{epoch}/{args.num_epochs}]")
        progress_bar.set_postfix(loss=f"{loss:.3f}")

    epoch_loss = running_loss / len(train_dataloader)
    # save
    save_checkpoint(args.output_dir, epoch, model, optimizer, scheduler, scaler, total_iter)
    return epoch_loss, total_iter

if __name__ == "__main__":
    args = parse_argument()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # test
    test_audio_prompt_path = 'data/LibriTTS/train-clean-100/6181/216552/6181_216552_000079_000002.wav'
    # test_text_prompt is 6181_216552_000079_000002.normalized.txt
    test_text_prompt = 'The individual passes away, society is deathless.'
    test_text = 'To be immortal is, precisely, not to suffer death.'

    # tensorboard
    tb_subsets = ['train', 'val']
    logger.init(args.output_dir, tb_subsets=tb_subsets)
    
    device = get_device()
    # dataset
    train_dataloader, val_dataloader = load_datasets(args)
    # model
    model = VALLE(
        d_model=args.decoder_dim, 
        nhead=args.num_heads, 
        num_layers=args.num_decoder_layers, 
        prefix_mode=1)
    
    # loss
    loss_criterion = compute_loss(device)
    # optimizer
    optimizer = get_optimizer(model, args.optimizer_type, args.learning_rate)
    # scheduler
    scheduler = get_scheduler(optimizer, args.scheduler_type, args.warmup_steps)
    # scaler
    scaler = get_scaler(args.scaler_type, args.dtype)
    
    # check previous chekpoints
    checkpoint, last_epoch = load_checkpoint(args.output_dir, model)
    model.to(device)

    total_iter = 0
    if checkpoint:
        def load(name, obj):
            s = checkpoint.get(name, None)
            if obj and s:
                obj.load_state_dict(s)
                checkpoint.pop(name)

        load("optimizer", optimizer)
        load("scheduler", scheduler)
        load("grad_scaler", scaler)
        total_iter = checkpoint["total_iter"]
    last_epoch = last_epoch if last_epoch else 0

    # autocast_type
    dtype, enabled = get_autocast_type(args.dtype)

    # training loop
    for epoch in range(last_epoch, args.num_epochs):
        if args.scheduler_type == 'Eden': scheduler.step_epoch(epoch - 1)

        train_loss, total_iter = train_one_epoch(
            train_dataloader,
            model,
            optimizer, scheduler, scaler,
            args,
            epoch, 
            total_iter,
            dtype, enabled)
        # evaluate after each epoch
        val_loss = evaluate(model, val_dataloader, loss_criterion, total_iter, dtype)

        # synthesize
        synthesized_audio = synthesize(model, test_audio_prompt_path, test_text_prompt, test_text, args.unique_text_tokens, device)

        model.train()

        ##############################  logs ##################################################
        logger.log(epoch, subset='train', data=OrderedDict([ ('epoch/loss', train_loss)]))
        logger.log(epoch, subset='val', data=OrderedDict([ ('epoch/loss', val_loss)]))
        # spectrogram
        syn_mel_specgram = get_mel_specgram(synthesized_audio, 24000)
        logger.log_image(epoch, subset='val',
                         data=OrderedDict([
                             ('spectrogram/synthesized', syn_mel_specgram),
                             ]))
        # audio
        prompt_audio = read_audio_waveform(test_audio_prompt_path)
        logger.log_audio(epoch, 24000, subset='val',
                         data=OrderedDict([
                             ('audio/prompt', prompt_audio),
                             ('audio/synthesized', synthesized_audio),
                             ]))
        # logger.log_text(epoch, subset='train', data=OrderedDict([('text', audio_text)]))

    logger.flush()
    print('inished traning')
