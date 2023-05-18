import argparse
import glob
from pathlib import Path

import sox
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--suffix", default=".qnt.pt")
    args = parser.parse_args()

    paths = [*args.folder.rglob(f"*{args.suffix}")]

    csv_file = open(str(args.folder)+'/metadata.csv', "w", encoding='utf8')
    for path in tqdm(paths):
        audio_file_path = str(path)
        phone_file_path = audio_file_path.replace('.qnt.pt', '.phn.txt')
        speaker_id = audio_file_path.split('/')[-1].split('_')[0]
        duration = sox.file_info.duration(audio_file_path.replace('.qnt.pt', '.wav'))
        write_line =  audio_file_path + '|' + phone_file_path + '|' + str(speaker_id) + '|' + str(duration) + '\n'
        csv_file.write(write_line)
    csv_file.close()

