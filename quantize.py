import os
import openvino as ov
from transformers import AutoConfig, AutoTokenizer
import nncf
from pathlib import Path
import argparse
import shutil
from utils.gptq import is_gptq

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    default='fp16_model',
                    required=False,
                    type=str,
                    help='orignal model path')
parser.add_argument('-o',
                    '--output',
                    default='./compressed_model',
                    required=False,
                    type=str,
                    help='Required. path to save the int4 ir model')
parser.add_argument('-p',
                    '--precision',
                    required=False,
                    default="int4",
                    type=str,
                    choices=["int8", "int4"],
                    help='int8 or int4')
args = parser.parse_args()


compressed_model_dir = Path(args.output)
orignal_model_dir = Path(args.model_id)
if compressed_model_dir.exists() == False:
    os.mkdir(compressed_model_dir)

model_config = AutoConfig.from_pretrained(
    args.model_id, trust_remote_code=True)
gptq_applied = is_gptq(model_config)

print(" --- loading model --- ")
if orignal_model_dir.exists():
    print(" --- using local model --- ")
    ov_model = ov.Core().read_model(orignal_model_dir / "openvino_model.xml")
    print(" --- compressing model --- ")
    if args.precision == "int4" and not gptq_applied:
        print(" --- exporting int4 model --- ")
        compressed_model = nncf.compress_weights(
            ov_model, mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=0.8)
    elif args.precision == "int8" and not gptq_applied:
        print(" --- exporting int8 model --- ")
        compressed_model = nncf.compress_weights(ov_model)
    else:
        raise RuntimeError(
            "Can not quantize a GPTQ model"
        )
    ov.save_model(compressed_model, compressed_model_dir / "openvino_model.xml")
    shutil.copy(compressed_model_dir / 'config.json', compressed_model_dir / 'config.json')
else:
    print(" --- using remote model --- ")
    from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
    model_compression_params = {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        }
    if args.precision == "int4" and not gptq_applied:
        print(" --- exporting int4 model --- ")
        compressed_model = OVModelForCausalLM.from_pretrained(
        args.model_id, export=True, compile=False,
        quantization_config=OVWeightQuantizationConfig(bits=4, **model_compression_params)
    )
    elif args.precision == "int8" and not gptq_applied:
        print(" --- exporting int8 model --- ")
        compressed_model = OVModelForCausalLM.from_pretrained(
            args.model_id, export=True, compile=False, load_in_8bit=True
        )
    else:
        raise RuntimeError(
            "Can not quantize a GPTQ model"
        )
    compressed_model.save_pretrained(compressed_model_dir)

print(" --- exporting tokenizer --- ")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.save_pretrained(compressed_model_dir)