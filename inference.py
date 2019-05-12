import argparse
import matplotlib
import matplotlib.pylab as plt

import sys
import numpy as np
import torch
import time
import scipy
from waveglow import glow

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model, init_distributed
from text import text_to_sequence
from glove import get_word, create_glove_dict

def do_full_inference(checkpoint_path, text, encoder_conditioning=False):
    glove = create_glove_dict()
    #glove = {"unknown token": [i for i in range(0, 300)]}
    model = setup_model(checkpoint_path,encoder_conditioning)
    mel_outputs, mel_outputs_postnet, alignments = text_to_mel(model, text, glove)
    return (mel_outputs, mel_outputs_postnet, alignments)

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

def text_to_mel(model, text, glove):
    print("Running inference on text: {}".format(text))
    start_time = time.time()
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    input = sequence
    if model.encoder_conditioning:
        words = text.strip().split()
        words_v = [get_word(glove, word) for word in words]
        words_v = torch.FloatTensor(words_v)
        input = (sequence, words_v)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(input)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))
    print("Finished inference in {} seconds".format(time.time() - start_time))
    return (mel_outputs, mel_outputs_postnet, alignments)

def mel_to_audio(waveglow, mel):
    audio = waveglow.infer(mel, sigma=0.666)
    return audio

def write_audio(audio_data, file_name):
    scipy.io.wavfile.write(file_name, 22050, audio_data)

def get_waveglow():
    import sys
    sys.path.insert(0, './waveglow')
    waveglow = torch.load('waveglow_256channels.pt')['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    return waveglow

def setup_model(checkpoint_path, encoder_conditioning=False):
    print("Loading Model from checkpoint {}".format(checkpoint_path))
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark =  True
    hparams = create_hparams()
    hparams.encoder_conditioning = encoder_conditioning
    hparams.fp16_run = True
    hparams.distributed_run = True
    init_distributed(hparams, 1, 0, 'group_name')
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    return model


def parsing_stuff_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default='output_full_tacotron_ed/checkpoint_4500',
                        required=False, help='checkpoint path')
    parser.add_argument('-t', '--text', type=str, default='this is a test string one two three',
                        required=False, help='text to pass through tacotron')
    parser.add_argument('-e', '--encoder_conditioning', action='store_true')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    args = parser.parse_args()
    do_full_inference(args.checkpoint_path, args.text, args.encoder_conditioning)

    
