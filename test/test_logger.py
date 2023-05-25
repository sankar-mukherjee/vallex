
# import sys
# sys.path.append('utils')

from collections import OrderedDict

import utils.logger as logger
from vallex.utils import get_mel_specgram, read_audio_waveform

if __name__ == "__main__":

    output_dir = 'output/test'
    tb_subsets = ['train', 'val']
    test_audio_prompt_path = 'data/LibriTTS/train-clean-100/6181/216552/6181_216552_000079_000002.wav'
    prompt_audio = read_audio_waveform(test_audio_prompt_path)
    mel_specgram = get_mel_specgram(prompt_audio, 24000)

    logger.init(output_dir, tb_subsets=tb_subsets)

    for i in range(100):
        logger.log(i, subset='train',
                        data=OrderedDict([
                            ('loss/l', 0.5-i/10),
                            ('loss/u', 0.5-i/10),
                            ('took', 0.5-i/10),
                            ]))
        logger.log(i, subset='val',
                        data=OrderedDict([
                            ('loss/l', 0.2-i/10),
                            ('loss/u', 0.5-i/10),
                            ('took', 0.5-i/10),
                            ]))

    for epoch in range(3):
        # audio
        logger.log_audio(epoch, 24000, subset='train',
                            data=OrderedDict([
                                ('audio/prompt', prompt_audio),
                                ('audio/synthesized', prompt_audio),
                                ]))

        logger.log_audio(epoch, 24000, subset='val',
                            data=OrderedDict([
                                ('audio/prompt', prompt_audio),
                                ('audio/synthesized', prompt_audio),
                                ]))
        # spectrogram
        logger.log_image(epoch, subset='train',
                         data=OrderedDict([
                             ('spectrogram/synthesized', mel_specgram),
                             ]))
        # text
        logger.log_text(epoch, subset='train',
                         data=OrderedDict([
                             ('text', 'huha'),
                             ]))
