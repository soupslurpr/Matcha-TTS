import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime.quantization
import torch
from lightning import LightningModule
from onnxruntime.quantization import quantize_dynamic, QuantType
from torch import Tensor
from torch.export import Dim

from matcha.cli import VOCODER_URLS, load_matcha, load_vocoder
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.model import fix_len_compatibility, sequence_mask, generate_path, denormalize

# TODO: update to a newer opset
DEFAULT_OPSET = 21

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MatchaEncoder(LightningModule):
    def __init__(self, matcha: MatchaTTS):
        super().__init__()
        self.n_spks = matcha.n_spks
        if self.n_spks > 1:
            self.spk_emb = matcha.spk_emb
        self.encoder = matcha.encoder

    def forward(self, x: Tensor, x_lengths: Tensor, length_scale: Tensor, spks: Tensor = None):
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        if spks is None:
            return y_lengths, mu_y, y_mask
        else:
            return y_lengths, mu_y, y_mask, spks


def get_exportable_encoder(encoder: MatchaEncoder):
    """
    Return an appropriate `LighteningModule` and output-node names
    """

    model, output_names = encoder, ["y_lengths", "mu_y", "y_mask", "spks"]
    return model, output_names


def get_encoder_inputs(is_multi_speaker: bool):
    """
    Create dummy encoder inputs for tracing
    """
    dummy_input_length = 5000
    x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
    x_lengths = torch.LongTensor([dummy_input_length])
    length_scale = torch.FloatTensor([1.0])

    model_inputs = [x, x_lengths, length_scale]
    input_names = [
        "x",
        "x_lengths",
        "length_scale",
    ]

    if is_multi_speaker:
        spks = torch.LongTensor([1])
        model_inputs.append(spks)
        input_names.append("spks")

    return tuple(model_inputs), input_names


class MatchaDecoder(LightningModule):
    def __init__(self, matcha: MatchaTTS):
        super().__init__()
        # self.n_spks = matcha.n_spks
        # if self.n_spks > 1:
        #     self.spk_emb = matcha.spk_emb
        self.decoder = matcha.decoder
        self.mel_mean = matcha.mel_mean
        self.mel_std = matcha.mel_std

        self.vocoder, _ = load_vocoder("generator_v3", "hifi-gan/upstream-trained-models/LJ_V3/generator_v3", "cpu")

    def forward(self, mu_y: Tensor, y_mask: Tensor, n_timesteps: Tensor, temperature: Tensor, spks: Tensor = None):
        # TODO: if n_timesteps can't be dynamic, move the forward parameter to __init__ to make it static
        n_timesteps = torch.tensor(5)

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)

        mel = denormalize(decoder_outputs, self.mel_mean, self.mel_std)

        wavs = self.vocoder(mel).clamp(-1, 1)
        return wavs.squeeze(1)

        # return denormalize(decoder_outputs, self.mel_mean, self.mel_std)


def get_exportable_decoder(decoder: MatchaDecoder):
    """
    Return an appropriate `LighteningModule` and output-node names
    """

    model, output_names = decoder, ["mel"]
    return model, output_names


def get_decoder_inputs(out_size: int, is_multi_speaker: bool):
    """
    Create dummy decoder inputs for tracing
    """
    dummy_input_length = out_size
    mu_y = torch.rand(size=[1, 80, dummy_input_length])
    y_mask = torch.rand(size=[1, 1, dummy_input_length])
    n_timesteps = torch.tensor(5)
    temperature = torch.tensor(0.667)

    model_inputs = [mu_y, y_mask, n_timesteps, temperature]
    input_names = [
        "mu_y",
        "y_mask",
        "n_timesteps",
        "temperature",
    ]

    if is_multi_speaker:
        spks = torch.LongTensor([1])
        model_inputs.append(spks)
        input_names.append("spks")

    return tuple(model_inputs), input_names


def main():
    parser = argparse.ArgumentParser(description="Export 🍵 Matcha-TTS to ONNX")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument("output", type=str, help="Path to output folder")
    parser.add_argument(
        "--n-timesteps", type=int, default=5, help="Number of steps to use for reverse diffusion in decoder (default 5)"
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version to use (default 21")

    args = parser.parse_args()

    print(f"[🍵] Loading Matcha checkpoint from {args.checkpoint_path}")
    print(f"Setting n_timesteps to {args.n_timesteps}")

    checkpoint_path = Path(args.checkpoint_path)
    matcha = load_matcha(checkpoint_path.stem, checkpoint_path, "cpu")

    is_multi_speaker = matcha.n_spks > 1

    encoder_dummy_input, encoder_input_names = get_encoder_inputs(is_multi_speaker)
    encoder, encoder_output_names = get_exportable_encoder(MatchaEncoder(matcha))
    encoder = encoder.eval()

    decoder_dummy_input, decoder_input_names = get_decoder_inputs(matcha.out_size, is_multi_speaker)
    # TODO: if n_timesteps needs to be static, make it a parameter to MatchaDecoder
    decoder, decoder_output_names = get_exportable_decoder(MatchaDecoder(matcha))
    decoder = decoder.eval()

    # Create the output directory (if not exists)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # static batch size for exported model
    batch_size = Dim.STATIC
    max_text_length = Dim("max_text_length")
    length_scale = Dim.STATIC
    encoder_dynamic_shapes = {
        "x": {0: batch_size, 1: max_text_length},
        "x_lengths": {0: batch_size},
        "length_scale": {0: length_scale},
    }

    y_max_length_dim = Dim.STATIC
    decoder_dynamic_shapes = {
        "mu_y": {0: batch_size, 2: y_max_length_dim},
        "y_mask": {0: batch_size, 2: y_max_length_dim},
        "n_timesteps": {},
        "temperature": {},
    }

    with torch.inference_mode():
        initial_export_directory = f"{args.output}/initial"
        Path(initial_export_directory).mkdir(parents=True, exist_ok=True)
        initial_encoder_export_path = f"{initial_export_directory}/encoder.onnx"
        initial_decoder_export_path = f"{initial_export_directory}/decoder.onnx"
        encoder.to_onnx(
            initial_encoder_export_path,
            encoder_dummy_input,
            input_names=encoder_input_names,
            output_names=encoder_output_names,
            dynamic_shapes=encoder_dynamic_shapes,
            opset_version=args.opset,
            export_params=True,
            dynamo=True,
            external_data=False,
        )
        decoder.to_onnx(
            initial_decoder_export_path,
            decoder_dummy_input,
            input_names=decoder_input_names,
            output_names=decoder_output_names,
            dynamic_shapes=decoder_dynamic_shapes,
            opset_version=args.opset,
            export_params=True,
            dynamo=True,
            external_data=False,
        )

        print(f"[🍵] Initial ONNX model exported to {initial_export_directory}")

        optimized_export_directory = f"{args.output}/optimized"
        Path(optimized_export_directory).mkdir(parents=True, exist_ok=True)
        optimized_encoder_export_path = f"{optimized_export_directory}/encoder.onnx"
        optimized_decoder_export_path = f"{optimized_export_directory}/decoder.onnx"
        onnxruntime.quantization.quant_pre_process(
            initial_encoder_export_path,
            optimized_encoder_export_path,
            # TODO: Encoder needs `--skip_symbolic_shape true` for some reason.
            skip_symbolic_shape=True,
        )
        onnxruntime.quantization.quant_pre_process(
            initial_decoder_export_path,
            optimized_decoder_export_path,
        )

        print(f"[🍵] Optimized ONNX model exported to {optimized_export_directory}")

        quantized_export_directory = f"{args.output}/quantized"
        Path(quantized_export_directory).mkdir(parents=True, exist_ok=True)
        quantized_encoder_export_path = f"{quantized_export_directory}/encoder.onnx"
        quantized_decoder_export_path = f"{quantized_export_directory}/decoder.onnx"
        quantize_dynamic(
            optimized_encoder_export_path,
            quantized_encoder_export_path,
            # exclude Conv since ONNX Runtime can't run the quantized version of it yet
            op_types_to_quantize=['MatMul', 'Attention', 'LSTM'],
            weight_type=QuantType.QInt8,
        )
        quantize_dynamic(
            optimized_decoder_export_path,
            quantized_decoder_export_path,
            # exclude Conv since ONNX Runtime can't run the quantized version of it yet
            op_types_to_quantize=['MatMul', 'Attention', 'LSTM'],
            weight_type=QuantType.QInt8,
        )

        print(f"[🍵] Quantized ONNX model exported to {quantized_export_directory}")

        optimized_quantized_export_directory = f"{args.output}/optimized_quantized"
        Path(optimized_quantized_export_directory).mkdir(parents=True, exist_ok=True)
        optimized_quantized_encoder_export_path = f"{optimized_quantized_export_directory}/encoder.onnx"
        optimized_quantized_decoder_export_path = f"{optimized_quantized_export_directory}/decoder.onnx"
        onnxruntime.quantization.quant_pre_process(
            quantized_encoder_export_path,
            optimized_quantized_encoder_export_path,
            # TODO: Quantized encoder optimization needs skip_symbolic_shape=True for some reason.
            skip_symbolic_shape=True,
        )
        onnxruntime.quantization.quant_pre_process(
            quantized_decoder_export_path,
            optimized_quantized_decoder_export_path,
            # TODO: Quantized decoder optimization now needs skip_symbolic_shape=True too for some reason.
            skip_symbolic_shape=True,
        )

        print(f"[🍵] Optimized quantized ONNX model exported to {optimized_quantized_export_directory}")

        print(f"[🍵] ONNX models exported to {args.output}")

        print(f"""[🍵] Try inferencing by running `python -m matcha.onnx.infer --text "The quick brown fox jumps """ +
              f"""over the lazy dog." --output-dir outputs {optimized_quantized_export_directory}`""")


if __name__ == "__main__":
    main()
