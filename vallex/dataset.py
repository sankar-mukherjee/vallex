import random
from functools import cached_property
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from vallex.utils import to_gpu
from embeddings.tokenizer import TextTokenizer, tokenize_text
from embeddings.collation import get_text_token_collater


def load_filepaths_and_text(metadata_csv, split="|"):
    def split_line(line):
        parts = line.strip().split(split)
        paths, non_paths = parts[:-2], parts[-2:]
        return tuple(str(Path(p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        fpaths_and_text += [split_line(line) for line in f]
    return fpaths_and_text

# Define your dataset and dataloader
class TTSDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            metadata_csv,
            unique_text_tokens,
            min_duration,
            max_duration,
        ):
        self.data_dir = data_dir
        self.audio_and_text_paths = load_filepaths_and_text(metadata_csv)
        self.audio_and_text_paths = list(filter(lambda x: 
                                                (min_duration < float(x[-1]) and float(x[-1]) > max_duration), 
                                                self.audio_and_text_paths))

        self.text_tokenizer = TextTokenizer()
        self.text_collater = get_text_token_collater(unique_text_tokens)
    
    def __getitem__(self, index):
        audiopath, text, speaker, *_ = self.audio_and_text_paths[index]
        audio_embs = torch.load(audiopath)[0].t()
        phonemes = tokenize_text(self.text_tokenizer, text=text)
        text_emb, _ = self.text_collater([phonemes])
        
        return audio_embs, text_emb.t(), torch.tensor(int(speaker))
    
    def __len__(self):
        return len(self.audio_and_text_paths)


def collate_fn(batch):
    # Separate the inputs and lengths
    audios, texts, speakers = zip(*batch)
    
    # Pad the text inputs to the maximum length in the batch
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_texts = torch.squeeze(padded_texts)

    # Pad the audio inputs to the maximum length in the batch
    padded_audios = pad_sequence(audios, batch_first=True)
    
    # Calculate the lengths of each sample
    text_lens = torch.tensor([len(text) for text in texts])
    
    # Calculate the lengths of each sample
    audio_lens = torch.tensor([audio.shape[0] for audio in audios])

    padded_audios = to_gpu(padded_audios)
    audio_lens = to_gpu(audio_lens)
    padded_texts = to_gpu(padded_texts)
    text_lens = to_gpu(text_lens)
    speakers = to_gpu(torch.stack(speakers))

    return padded_audios, audio_lens, padded_texts, text_lens, speakers
