import torch
from torch.utils.data import  Dataset
import random
from tqdm import tqdm
from pathlib import Path
from functools import cached_property
from torch.nn.utils.rnn import pad_sequence
from vallex.utils import to_gpu


def load_filepaths_and_text(metadata_csv, split="|"):
    def split_line(line):
        parts = line.strip().split(split)
        paths, non_paths = parts[:-2], parts[-2:]
        return tuple(str(Path(p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        fpaths_and_text += [split_line(line) for line in f]
    return fpaths_and_text


def _get_phones(path):
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
    return ["<s>"] + content.split() + ["</s>"]

# Define your dataset and dataloader
class TTSDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            metadata_csv,
            min_duration,
            max_duration,
        ):
        self.data_dir = data_dir
        self.audio_and_text_paths = load_filepaths_and_text(metadata_csv)
        self.audio_and_text_paths = list(filter(lambda x: 
                                                (min_duration < float(x[-1]) and float(x[-1]) > max_duration), 
                                                self.audio_and_text_paths))

        self.phone_symmap = self._get_phone_symmap()

    def _get_phone_symmap(self):
        # Note that we use phone symmap starting from 1 so that we can safely pad 0.
        return {s: i for i, s in enumerate(self.phones, 1)}
    
    @cached_property
    def phones(self):
        return sorted(set().union(*[_get_phones(path[1]) for path in self.audio_and_text_paths]))
    
    def __getitem__(self, index):
        audiopath, textpath, speaker, *_ = self.audio_and_text_paths[index]
        audio_embs = torch.load(audiopath)[0].t()
        text_emb = torch.tensor([*map(self.phone_symmap.get, _get_phones(textpath))])

        return audio_embs, text_emb, torch.tensor(int(speaker))
    
    def __len__(self):
        return len(self.audio_and_text_paths)


def collate_fn(batch):
    # Separate the inputs and lengths
    audios, texts, speakers = zip(*batch)
    
    # Pad the text inputs to the maximum length in the batch
    padded_texts = pad_sequence(texts, batch_first=True)
    
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
