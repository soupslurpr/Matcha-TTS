This is GrapheneOS's fork of 🍵 Matcha-TTS [ICASSP 2024] (https://arxiv.org/abs/2309.03199) with major speed and 
efficiency improvements compared to the original.

The original code and README.md can be found at https://github.com/shivammehta25/Matcha-TTS.

Please use Python 3.11 for this repository.

🍵 Matcha-TTS is a new approach to non-autoregressive neural TTS that uses 
[conditional flow matching](https://arxiv.org/abs/2210.02747) (similar to 
[rectified flows](https://arxiv.org/abs/2209.03003)) to speed up ODE-based speech synthesis:

- Is probabilistic
- Has compact memory footprint
- Sounds highly natural
- Is very fast to synthesise from

This fork adds major speed and efficiency improvements to training and inference.

Training speed is improved through aligning the out_size chunk to a multiplier that starts from the beginning. The 
out_size is set to 64 to align with the mel spectrogram "borders". 

During inference, the decoder should be run in chunks of out_size (default is 64). This way, you can decode in a 
streaming fashion to provide fast time-to-first-audio during inference. This is especially useful for edge 
use-cases which require low latency such as an on-device text-to-speech engine.

Parameter size has been significantly reduced from 18.2 million to 4.7 million, meaning it's only around 1/4 the 
parameters of the original! This significantly increases training and inference speed and is achieved through training 
with precomputed durations, setting prior_loss to false, and the out_size alignment previously mentioned.

Setting prior_loss to false means that the encoder does not have its own loss which shapes its output to the target 
mel spectrogram. Instead, it uses the same loss as the decoder. This results in much more efficient encoding and 
therefore allows significantly reducing parameter size. The catch is, that the model can't learn durations properly 
with no prior loss, but, we can train a model with prior loss, compute durations, then train again with precomputed 
durations without prior loss.

## Training

Currently, only training on LJSpeech has been tested and used. Follow the directions below to train on LJSpeech.

1. Clone and enter this repository.

```commandline
cd Matcha-TTS
```

2. Clone our fork of Misaki and install it using `pip install -e ../misaki`. Make sure to also clone graphemes_to_phonemes 
and have it contain models as our fork of Misaki depends on it.

2. Download the dataset from [here](https://keithito.com/LJ-Speech-Dataset/), extract it to `data/LJSpeech-1.1`, and 
prepare the file lists to point to the extracted data like for 
[item 5 in the setup of the NVIDIA Tacotron 2 repo](https://github.com/NVIDIA/tacotron2#setup).

3. Install this package from source

```commandline
pip install -e .
```

4. Go to `configs/data/ljspeech.yaml` and change to the paths of your train and validation filelists

The default is the names of the files used by the NVIDIA Tacotron 2 repo; you just need to download them from 
https://github.com/NVIDIA/tacotron2/tree/master/filelists, rename by removing "_filelist" at the end before ".txt", and 
place them at the correct paths.

```yaml
train_filelist_path: data/filelists/ljs_audio_text_train.txt
valid_filelist_path: data/filelists/ljs_audio_text_val.txt
```

5. Generate normalisation statistics with the yaml file of dataset configuration

```bash
matcha-data-stats -i ljspeech.yaml
# Output:
#{'mel_mean': -5.51702880859375, 'mel_std': 2.064393997192383}
```

Update these values in `configs/data/ljspeech.yaml` under `data_statistics` key.

```bash
data_statistics:  # Computed for ljspeech dataset
  mel_mean: -5.51702880859375
  mel_std: 2.064393997192383
```

6. Run initial training to compute durations

```commandline
python matcha/train.py experiment=ljspeech
```

or for multi-gpu training, run

```bash
python matcha/train.py experiment=ljspeech trainer.devices=[0,1]
```

TODO() steps seems to be a good stopping point.

7. Synthesise from the initial custom trained model

Make sure that the initial model works at least OK, the quality will get better in the final model.

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT>
```

8. Generate durations

Follow the section [extract phoneme alignments from Matcha-TTS](#Extract-phoneme-alignments-from-Matcha-TTS) and put the
durations inside the `data/LJSpeech-1.1/durations` directory.

7. Run final training with precomputed durations

```commandline
python matcha/train.py experiment=ljspeech_from_durations
```

or for multi-gpu training, run

```bash
python matcha/train.py experiment=ljspeech_from_durations trainer.devices=[0,1]
```

TODO() steps seems to be a good stopping point.

9. Synthesize from the final custom trained model

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT>
```

## ONNX support

### ONNX export

To export a checkpoint to ONNX, first install ONNX Runtime with

```bash
pip install onnxruntime
```

then run the following:

```bash
python3 -m matcha.onnx.export matcha.ckpt onnx_model_folder --n-timesteps 5
```

**Note** that `n_timesteps` is treated as a hyper-parameter rather than a model input. This means you should specify it during export (not during inference). If not specified, `n_timesteps` is set to **5**.

### ONNX inference

To run inference on the exported model, first install `onnxruntime` using

```bash
pip install onnxruntime
pip install onnxruntime-gpu  # for GPU inference
```

then use the following:

```bash
python3 -m matcha.onnx.infer onnx_model_folder --text "hey" --output-dir ./outputs
```

This will write `.wav` audio files to the output directory.

You can also control synthesis parameters:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --temperature 0.4 --speaking_rate 0.9 --spk 0
```

To run inference on **GPU**, make sure to install **onnxruntime-gpu** package, and then pass `--gpu` to the inference command:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --gpu
```

## Extract phoneme alignments from Matcha-TTS

If the dataset is structured as

```bash
data/
└── LJSpeech-1.1
    ├── metadata.csv
    ├── README
    ├── test.txt
    ├── train.txt
    ├── val.txt
    └── wavs
```
Then you can extract the phoneme level alignments from a Trained Matcha-TTS model using:
```bash
python  matcha/utils/get_durations_from_trained_model.py -i dataset_yaml -c <checkpoint>
```
Example:
```bash
python  matcha/utils/get_durations_from_trained_model.py -i ljspeech.yaml -c matcha_ljspeech.ckpt
```
or simply:
```bash
matcha-tts-get-durations -i ljspeech.yaml -c matcha_ljspeech.ckpt
```
---
## Train using extracted alignments

In the datasetconfig turn on load duration.
Example: `ljspeech.yaml`
```
load_durations: True
```
or see an examples in configs/experiment/ljspeech_from_durations.yaml
