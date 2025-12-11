import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType


def validate_args(args):
    assert args.input, "Input ONNX model must be provided"
    assert args.output, "Output quantized ONNX model path must be provided"
    return args


def main():
    parser = argparse.ArgumentParser(description="Quantize 🍵 Matcha-TTS ONNX model")
    parser.add_argument(
        "--input",
        type=str,
        help="ONNX model folder to quantize",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="quantized ONNX model folder path",
    )

    args = parser.parse_args()
    args = validate_args(args)


    quantize_dynamic(
        f"{args.input}/encoder.onnx",
        f"{args.output}/encoder.onnx",
        # exclude Conv since ONNX Runtime can't run the quantized version of it yet
        op_types_to_quantize=['MatMul', 'Attention', 'LSTM'],
        weight_type=QuantType.QInt8,
    )
    quantize_dynamic(
        f"{args.input}/decoder.onnx",
        f"{args.output}/decoder.onnx",
        # exclude Conv since ONNX Runtime can't run the quantized version of it yet
        op_types_to_quantize=['MatMul', 'Attention', 'LSTM'],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
