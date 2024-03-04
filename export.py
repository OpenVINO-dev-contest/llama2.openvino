import os
import openvino as ov
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import argparse
import warnings
try:
    from optimum.exporters.openvino.stateful import make_stateful
    from optimum.exporters.openvino.stateful import fuse_cache_reorder
except ImportError:
    warnings.warn(
        "We recommend to update optimum-intel for getting optimal performance")
    make_stateful = None
    fuse_cache_reorder = None


def patch_stateful(ov_model, model_type):
    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 1 if model_type == "chatglm" else 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs,
                       key_value_input_names, batch_dim)
    make_stateful(
        ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim, num_attention_heads, None
    )


def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        default='meta-llama/Llama-2-7b-hf',
                        required=False,
                        type=str,
                        help='orignal model path')
    parser.add_argument('-o',
                        '--output',
                        default='./fp16_model',
                        required=False,
                        type=str,
                        help='Required. path to save the ir model')
    args = parser.parse_args()

    ir_model_path = Path(args.output)
    if ir_model_path.exists() == False:
        os.mkdir(ir_model_path)
    ir_model_file = ir_model_path / "openvino_model.xml"

    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True)

    pt_model.config.save_pretrained(ir_model_path)
    pt_model.config.use_cache = True
    outs = pt_model(input_ids=torch.ones((2, 10), dtype=torch.long))
    inputs = ["input_ids"]
    outputs = ["logits"]

    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
    }
    inputs += ["attention_mask", "position_ids"]
    for idx in range(len(outs.past_key_values)):
        inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        dynamic_shapes[inputs[-1]] = {0: "batch_size", 2: "past_sequence + sequence"}
        dynamic_shapes[inputs[-2]] = {0: "batch_size", 2: "past_sequence + sequence"}
        outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

    dummy_inputs = {
        "input_ids": torch.ones((2, 2), dtype=torch.long),
        "attention_mask": torch.ones((2, 12), dtype=torch.long),
        "position_ids": torch.tensor([[10, 11], [10, 11]], dtype=torch.long),
        "past_key_values": outs.past_key_values,
    }
    pt_model.config.torchscript = True
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    for inp_name, m_input, input_data in zip(
        inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())
    ):
        input_node = m_input.get_node()
        if input_node.element_type == ov.Type.dynamic:
            m_input.get_node().set_element_type(ov.Type.f32)
        shape = list(input_data.shape)
        if inp_name in dynamic_shapes:
            for k in dynamic_shapes[inp_name]:
                shape[k] = -1
        input_node.set_partial_shape(ov.PartialShape(shape))
        m_input.get_tensor().set_names({inp_name})

    for out, out_name in zip(ov_model.outputs, outputs):
        out.get_tensor().set_names({out_name})

    ov_model.validate_nodes_and_infer_types()
    if make_stateful is not None:
        patch_stateful(ov_model, "llama2")
    ov.save_model(ov_model, ir_model_file)
    del ov_model
    del pt_model

    print(" --- exporting tokenizer --- ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(ir_model_path)
