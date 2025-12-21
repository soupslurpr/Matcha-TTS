import argparse
import os
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from numpy.core.records import ndarray

from matcha.cli import plot_spectrogram_to_numpy, process_text


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.speaking_rate >= 0, "Speaking rate must be greater than 0"
    return args


def write_wavs(model, inputs, output_dir, external_vocoder=None):
    if external_vocoder is None:
        print("The provided model has the vocoder embedded in the graph.\nGenerating waveform directly")
        t0 = perf_counter()
        wavs, wav_lengths = model.run(None, inputs)
        infer_secs = perf_counter() - t0
        mel_infer_secs = vocoder_infer_secs = None
    else:
        print("[🍵] Generating mel using Matcha")
        mel_t0 = perf_counter()
        mels, mel_lengths = model.run(None, inputs)
        mel_infer_secs = perf_counter() - mel_t0
        print("Generating waveform from mel using external vocoder")
        vocoder_inputs = {external_vocoder.get_inputs()[0].name: mels}
        vocoder_t0 = perf_counter()
        wavs = external_vocoder.run(None, vocoder_inputs)[0]
        vocoder_infer_secs = perf_counter() - vocoder_t0
        wavs = wavs.squeeze(1)
        wav_lengths = mel_lengths * 256
        infer_secs = mel_infer_secs + vocoder_infer_secs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (wav, wav_length) in enumerate(zip(wavs, wav_lengths)):
        output_filename = output_dir.joinpath(f"output_{i + 1}.wav")
        audio = wav[:wav_length]
        print(f"Writing audio to {output_filename}")
        sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if mel_infer_secs is not None:
        mel_rtf = mel_infer_secs / wav_secs
        print(f"Matcha RTF: {mel_rtf}")
    if vocoder_infer_secs is not None:
        vocoder_rtf = vocoder_infer_secs / wav_secs
        print(f"Vocoder RTF: {vocoder_rtf}")
    print(f"Overall RTF: {rtf}")


def write_mels(model, inputs, output_dir):
    t0 = perf_counter()
    mels, mel_lengths = model.run(None, inputs)
    infer_secs = perf_counter() - t0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mel in enumerate(mels):
        output_stem = output_dir.joinpath(f"output_{i + 1}")
        plot_spectrogram_to_numpy(mel.squeeze(), output_stem.with_suffix(".png"))
        np.save(output_stem.with_suffix(".numpy"), mel)

    wav_secs = (mel_lengths * 256).sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"RTF: {rtf}")


def main():
    # print("Infer hasn't been updated to support splitting encoder according to decoder out_size yet.")
    # return
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "model",
        type=str,
        help="ONNX model folder to use",
    )
    parser.add_argument("--vocoder", type=str, default=None, help="Vocoder to use (defaults to None)")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--out-size", type=int, default=None, help="out_size to use, should be same as out_size of decoder (defaults to using full out_size)")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )

    args = parser.parse_args()
    args = validate_args(args)

    if args.gpu:
        providers = ["GPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    encoder = ort.InferenceSession(f"{args.model}/encoder.onnx", providers=providers)
    decoder = ort.InferenceSession(f"{args.model}/decoder.onnx", providers=providers)

    encoder_parameters = encoder.get_inputs()

    if args.text:
        text_lines = args.text.splitlines()
    else:
        with open(args.file, encoding="utf-8") as file:
            text_lines = file.read().splitlines()

    processed_lines = [process_text(0, line, "cpu") for line in text_lines]
    x = [line["x"].squeeze() for line in processed_lines]
    # Pad
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x = x.detach().cpu().numpy()
    x_lengths = np.array([line["x_lengths"].item() for line in processed_lines], dtype=np.int64)
    encoder_inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "length_scale": np.array([args.speaking_rate], dtype=np.float32),
    }
    decoder_inputs = {
        # TODO: remove n_timesteps parameter if it needs to be static for the model to be exported
        "n_timesteps": np.array([5]),
        "temperature": np.array([args.temperature], dtype=np.float32),
    }
    is_multi_speaker = len(encoder_parameters) == 4
    if is_multi_speaker:
        if args.spk is None:
            args.spk = 0
            warn = "[!] Speaker ID not provided! Using speaker ID 0"
            warnings.warn(warn, UserWarning)
        spks = np.repeat(args.spk, x.shape[0]).astype(np.int64)
        encoder_inputs["spks"] = spks

    print("[🍵] Running Matcha encoder")
    encoder_t0 = perf_counter()
    if is_multi_speaker:
        y_lengths, mu_y, y_mask, spks = encoder.run(None, encoder_inputs)
        decoder_inputs["spks"] = spks
    else:
        y_lengths, mu_y, y_mask = encoder.run(None, encoder_inputs)
    wav_lengths = y_lengths * 256
    encoder_infer_secs = perf_counter() - encoder_t0
    print("Generating waveform using Matcha decoder (included vocoder)")

    # Split encoder output as done in MatchaTTS.synthesize() to emulate streaming sequentially.
    decoder_t0 = perf_counter()
    mu_y = torch.from_numpy(mu_y)
    y_mask = torch.from_numpy(y_mask)
    y_max_length_in_batch = args.out_size or mu_y.shape[2]
    y_length_padding = ((mu_y.shape[2] / y_max_length_in_batch).__ceil__() * y_max_length_in_batch) - mu_y.shape[2]
    mu_y = torch.nn.functional.pad(mu_y, (0, y_length_padding), value=0)
    y_mask = torch.nn.functional.pad(y_mask, (0, y_length_padding), value=1)
    mu_y = torch.cat(mu_y.split(y_max_length_in_batch, 2), 0)
    y_mask = torch.cat(y_mask.split(y_max_length_in_batch, 2), 0)

    wavs = list()
    for index in range(mu_y.shape[0]):
        decoder_inputs["mu_y"] = mu_y[index, :, :].unsqueeze(0).numpy()
        decoder_inputs["y_mask"] = y_mask[index, :, :].unsqueeze(0).numpy()
        wav = decoder.run(None, decoder_inputs)
        wavs.append(wav)

    decoder_infer_secs = perf_counter() - decoder_t0
    infer_secs = encoder_infer_secs + decoder_infer_secs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate([np.squeeze(wav)[:(y_max_length_in_batch * 256)] for wav in wavs])
    audio = audio[:wav_lengths[0]]
    output_filename = output_dir.joinpath(f"output.wav")
    print(f"Writing audio to {output_filename}")
    sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if encoder_infer_secs is not None:
        encoder_rtf = encoder_infer_secs / wav_secs
        print(f"Matcha encoder RTF: {encoder_rtf}")
    if decoder_infer_secs is not None:
        decoder_rtf = decoder_infer_secs / wav_secs
        print(f"Matcha decoder RTF: {decoder_rtf}")
    print(f"Overall RTF: {rtf}")


if __name__ == "__main__":
    main()
