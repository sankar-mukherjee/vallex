import argparse
import collections
import glob
from pathlib import Path

import numpy as np
import sox
from symbol_table import SymbolTable
from tokenizer import TextTokenizer, tokenize_text
from tqdm import tqdm

# import sys
# sys.path.append('embeddings')


def filter_speaker_list(lines, category_index, count):
    speakers = list(map(lambda line: line.strip().split('|')[category_index], lines))
    speakers_freq = collections.Counter(speakers).most_common()

    target_speakers = [i[0] for i in speakers_freq if i[1] >= count]
    target_lines = [line for line in lines if line.strip().split('|')[category_index] in target_speakers]
    print('no of speakers in train data: '+ str(len(target_speakers)))

    val_speakers = [i[0] for i in speakers_freq if i[1] < count]
    val_lines = [line for line in lines if line.strip().split('|')[category_index] in val_speakers]
    print('no of speakers in val data: '+ str(len(val_speakers)))

    return target_lines, target_speakers, val_lines, val_speakers

def _get_text(path):
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--suffix", type=str, default=".qnt.pt")
    parser.add_argument("--filter_type", type=str, default="sample_wise:10")

    args = parser.parse_args()

    paths = [*args.folder.rglob(f"*{args.suffix}")]

    csv_file = open(str(args.folder)+'/metadata.csv', "w", encoding='utf8')
    for path in tqdm(paths):
        audio_file_path = str(path)
        phone_file_path = audio_file_path.replace('.qnt.pt', '.normalized.txt')

        text = _get_text(phone_file_path)
        speaker_id = audio_file_path.split('/')[-1].split('_')[0]
        duration = sox.file_info.duration(audio_file_path.replace('.qnt.pt', '.wav'))

        write_line =  audio_file_path + '|' + text + '|' + str(speaker_id) + '|' + str(duration) + '\n'
        csv_file.write(write_line)
    csv_file.close()

    # unique symbols
    print('get unique symbols')
    with open(str(args.folder)+'/metadata.csv', 'r') as f:
        lines = f.readlines()
    f.close()

    text_tokenizer = TextTokenizer()
    unique_symbols = set()
    for line in tqdm(lines):
        text = line.split('|')[1]
        phonemes = tokenize_text(text_tokenizer, text=text)
        unique_symbols.update(list(phonemes))

    unique_phonemes = SymbolTable()
    for s in sorted(list(unique_symbols)):
        unique_phonemes.add(s)
    unique_phonemes_file = f"{args.folder}/unique_text_tokens.k2symbols"
    unique_phonemes.to_file(unique_phonemes_file)

    # train val
    print('get train and val data')
    with open(str(args.folder)+'/metadata.csv', 'r') as f:
        lines = f.readlines()
    f.close()

    filter_type, split_percent = args.filter_type.split(':')
    if filter_type == 'speaker_wise':
        # speaker filter
        min_no_samples_per_speaker = 10
        speaker_idx = 2
        train_lines, _, val_lines, _ = filter_speaker_list(lines, speaker_idx, min_no_samples_per_speaker)
    if filter_type == 'sample_wise':
        val_lines, train_lines = np.split(lines, [int(len(lines)*int(split_percent)/100),])

    csv_file = open(str(args.folder)+'/metadata_train.csv', "w", encoding='utf8')
    csv_file.writelines(train_lines)
    csv_file.close()

    csv_file = open(str(args.folder)+'/metadata_val.csv', "w", encoding='utf8')
    csv_file.writelines(val_lines)
    csv_file.close()

    #
    print('done!')
