# Tacotron 2 (without wavenet)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). 

This implementation includes **distributed** and **automatic mixed precision** support
and uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

Visit our [website] for audio samples using our published [Tacotron 2] and
[WaveGlow] models.

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths 
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Multi-GPU (distributed) Supervised learning
1. `python -m multiproc train.py --output_directory=outdir_fulltacotron --log_directory=logdir --hparams=distributed_run=True,fp16_run=True,training_files=filelists/david/labelled/train.txt,validation_files=filelists/david/labelled/val.txt`

## Multi-GPU (distributed) Supervised learning with encoder conditioning
1.  `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True,training_files=filelists/david/labelled/val.txt,validation_files=filelists/david/labelled/val.txt,encoder_conditioning=True`
2. `python -m multiproc train.py --output_directory=outdir_full_tacotron_ed --log_directory=logdir --hparams=distributed_run=True,fp16_run=True,training_files=filelists/david/labelled/train.txt,validation_files=filelists/david/labelled/val.txt,encoder_conditioning=True,make_new_encoder=True -c outdir_unsupervised/checkpoint_10000 --warm_start`

## Multi-GPU Unsupervised learning
1. `python -m multiproc train.py --output_directory=outdir_unsupervised --log_directory=logdir --hparams=distributed_run=True,fp16_run=True,unsupervised=True,training_files=filelists/david/unlabelled/train_list.txt,validation_files=filelists/david/unlabelled/val_list.txt `

## Multi-GPU Supervised encoder validation
1. ` python -m multiproc validate.py --output_directory=outdir_val --log_directory=logdir -c outdir_full_tacotron_ed/checkpoint_4500 --hparams=distributed_run=True,fp16_run=True,validation_files=filelists/david/labelled/val.txt,encoder_conditioning=True`

## Multi-GPU Unsupervised validation
1. ` python -m multiproc validate.py --output_directory=outdir_val --log_directory=logdir -c outdir_unsupervised/checkpoint_10200 --hparams=distributed_run=True,fp16_run=True,unsupervised=True,training_files=filelists/david/unlabelled/train_list.txt,validation_files=filelists/david/unlabelled/val_list.txt`

## Inference
1. `out = inference.do_full_inference("outdir_full_tacotron_ed/checkpoint_3000", "hello my name is david", True)`
2. `audio, mel_outputs, mel_outputs_postnet, alignments, model, glove, glow = inference.do_full_audio("hello my name is david",1)`
2. `python -m multiproc inference.py -c outdir_full_tacotron_ed/checkpoint_3000 -t "hello my name is david" -e`

### Main Files for Training
filelists/david/labelled/val.txt
filelists/david/labelled/train.txt

## Inference demo
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
