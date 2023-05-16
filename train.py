import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vallex.dataset import MyDataset
# from vallex.model import VALLE
from vallex.model2 import VALLE
import argparse
from pathlib import Path
from vallex.loss import compute_loss
from vallex.dataset import collate_fn

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

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_argument()

    dataset = MyDataset(args.data_dir, args.metadata_csv)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Define your training loop
    model = VALLE(args.decoder_dim, args.num_heads, args.num_decoder_layers)
    criterion = compute_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter()

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            audio_embs, audio_lens, text_embs, text_lens, speaker = data
            
            optimizer.zero_grad()
            
            codes, ar_logits, ar_targets, nar_logits, nar_targets = model(text_embs, text_lens, audio_embs, audio_lens)
            loss, loss_info = criterion(audio_lens, ar_logits, ar_targets, nar_logits, nar_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss))

    writer.close()
