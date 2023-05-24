import torch
from typing import Any, Dict, List, Optional, Union
import torch.nn as nn
from pathlib import Path
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from vallex.modules.optim import ScaledAdam
import matplotlib.pyplot as plt
import torchaudio
import librosa
import numpy as np

LRSchedulerType = object

def get_autocast_type(dtype):
    autocast_dtype, enabled = torch.float32, False
    if dtype in ["bfloat16", "bf16"]: autocast_dtype, enabled = torch.bfloat16, True
    elif dtype in ["float16", "fp16"]: autocast_dtype, enabled = torch.float16, True
    return autocast_dtype, enabled

def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return device

def get_optimizer(model, learning_rate):
    # optimizer = Optimizer.Adam(model.parameters(), lr=args.learning_rate)

    parameters_names = []
    parameters_names.append([
        name_param_pair[0] for name_param_pair in model.named_parameters()
    ])
    optimizer = ScaledAdam(model.parameters(), lr=learning_rate,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
    optimizer.zero_grad()
    return optimizer

def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x

def save_checkpoint(out_dir, epoch, model,optimizer, scheduler, scaler, total_iter):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"checkpoint_epoch-{epoch}.pt"
    print(f"Saving checkpoint to {filename}")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "total_iter": total_iter,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(
    output_dir: Path,
    model: nn.Module,
    strict: bool = False,
) -> Dict[str, Any]:
    
    checkpoint_paths = list(output_dir.rglob("checkpoint_epoch-*.pt"))
    if len(checkpoint_paths)>0:
        last_epoch = max(list(map(lambda checkpoint_path: int(checkpoint_path.stem.split('-')[-1]), checkpoint_paths)))
        filename = output_dir / f"checkpoint_epoch-{last_epoch}.pt"
    else:
        return None, None
    
    assert filename.is_file(), f"{filename} does not exist!"

    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    model.load_state_dict(checkpoint["model"], strict=strict)
    checkpoint.pop("model")
    return checkpoint, last_epoch

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)

# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token

def plot_spectrogram(spectrogram, fig_size=(6, 3)):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy().squeeze()
    else:
        spectrogram = spectrogram.squeeze()

    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    return fig

def read_audio_waveform(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] == 2:
        wav = wav[:1]
    return wav

def get_mel_specgram(waveform, sample_rate):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy().squeeze()
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    mel_specgram = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_specgram
