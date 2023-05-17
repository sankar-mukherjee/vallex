import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vallex.dataset import TTSDataset
# from vallex.model import VALLE
from vallex.model2 import VALLE
import argparse
from pathlib import Path
from vallex.loss import compute_loss
from vallex.dataset import collate_fn
from torch.cuda.amp import GradScaler

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
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dtype", type=str, default='torch.float16')

    parser.add_argument("--filter_min_duration", type=float, default=0.0)
    parser.add_argument("--filter_max_duration", type=float, default=100.0)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_argument()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True


    dataset = TTSDataset(
        args.data_dir, 
        args.metadata_csv,
        args.filter_min_duration,
        args.filter_max_duration,
        )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Define your training loop
    model = VALLE(args.decoder_dim, args.num_heads, args.num_decoder_layers).to(device)
    criterion = compute_loss(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    scaler = GradScaler(enabled=(args.dtype in ["fp16", "float16"]), init_scale=1.0)

    writer = SummaryWriter()

    dtype, enabled = torch.float32, False
    if args.dtype in ["bfloat16", "bf16"]: dtype, enabled = torch.bfloat16, True
    elif args.dtype in ["float16", "fp16"]: dtype, enabled = torch.float16, True

    is_training = True

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            audio_embs, audio_lens, text_embs, text_lens, speaker = data
            
            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                with torch.set_grad_enabled(is_training):
                    codes, ar_logits, ar_targets, nar_logits, nar_targets = model(text_embs, text_lens, audio_embs, audio_lens)
                    loss, loss_info = criterion(audio_lens, ar_logits, ar_targets, nar_logits, nar_targets)
                assert loss.requires_grad == is_training

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss
            print('Batch %d loss: %.3f' % (i + 1, running_loss))

        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss))

    writer.close()
