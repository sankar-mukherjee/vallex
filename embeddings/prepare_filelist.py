import glob
from tqdm import tqdm

if __name__ == "__main__":
    path = 'data/test'
    file_paths = glob.glob(path+'/*.qnt.pt')


    csv_file = open(path+'/metadata.csv', "w", encoding='utf8')
    for i,_ in enumerate(tqdm(file_paths)):
        audio_file_path = file_paths[i]
        phone_file_path = audio_file_path.replace('.qnt.pt', '.phn.txt')
        speaker_id = audio_file_path.split('/')[-1].split('_')[0]
        write_line =  audio_file_path + '|' + phone_file_path + '|' + str(speaker_id) + '\n'
        csv_file.write(write_line)
    csv_file.close()

