# llama2.openvino

This sample shows how to implement a llama-based model with OpenVINO runtime.

<img width="947" alt="MicrosoftTeams-image (2)" src="https://github.com/OpenVINO-dev-contest/llama2.openvino/assets/91237924/c210507f-1fb2-4c68-a8d9-dae945df07d3">


- Please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints, for more [information](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- Please notice this repository is only for a functional test and personal study.

## Requirements

- Linux, Windows
- Python >= 3.9.0
- CPU or GPU compatible with OpenVINO.
- RAM: >=16GB
- vRAM: >=8GB

## 1. Environment configuration

    $ python3 -m venv openvino_env

    $ source openvino_env/bin/activate

    $ python3 -m pip install --upgrade pip
    
    $ pip install wheel setuptools
    
    $ pip install -r requirements.txt

setup access Tokens

    $ huggingface-cli login --token hf_xxxxxxxxx


## 2. Q&A Pipeline

### 2.1 Export IR model

from Transformers:

    $ python3 export.py --model_id 'meta-llama/Llama-2-7b-hf' --output {your_path}/Llama-2-7b-hf

or from Optimum-Intel:

    $ python3 export_op.py --model_id 'meta-llama/Llama-2-7b-hf' --output {your_path}/Llama-2-7b-hf

or for #GPTQ model:

    $ python3 export_op.py --model_id 'TheBloke/Llama-2-7B-Chat-GPTQ' --output {your_path}/Llama-2-7B-Chat-GPTQ

**Parameters that can be selected**

* `--model_id` - path (absolute path) to be used from Huggngface_hub (https://huggingface.co/models) or the directory
  where the model is located.
* `--output` - the address where the converted model is saved
* If you have difficulty accessing `huggingface`, you can try to use `mirror-hf` to download

### 2.2.  (Optional) quantize local IR model with #int8 or #int4 weight**

    $ python3 quantize.py --model_id {your_path}/Llama-2-7b-hf --precision int4 --output {your_path}/Llama-2-7b-hf-int4

**Parameters that can be selected**

* `--model_id` - The path to the directory where the OpenVINO IR model is located.
* `--precision` - Quantization precision: int8 or int4.
* `--output` - Path to save the model.

For more information on quantization configuration, please refer to [weight compression](https://github.com/openvinotoolkit/nncf/blob/release_v270/docs/compression_algorithms/CompressWeights.md)

### 2.3 Run pipeline

[Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference):

    $ python3 pipeline/generate_op.py --model_id {your_path}/Llama-2-7b-hf-int4 --prompt "what is openvino ?" --device "CPU"

**Parameters that can be selected**

* `--model_id` - HuggingFace model id or path to the directory where the OpenVINO IR model is located.
* `--prompt` - Maximum size of output tokens.
* `--max_sequence_length` - Maximum size of output tokens.
* `--device` - The device to run inference on. e.g "CPU","GPU".

or Restructured pipeline:

    $ python3 pipeline/generate.py --model_path {your_path}/Llama-2-7b-hf-int4 --prompt "what is openvino ?" --device "CPU"

**Parameters that can be selected**

* `--model_path` - The path to the directory where the OpenVINO IR model is located.
* `--max_sequence_length` - Maximum size of output tokens.
* `--device` - The device to run inference on. e.g "CPU","GPU".

## 3. Interactive demo

### 3.1. Interactive Q&A demo with Gradio

    $ python3 demo/qa_gradio.py --model_id {your_path}/Llama-2-7b-hf-int4

### 3.2. Chatbot demo with Streamlit

    $ python3 quantize.py --model_id 'meta-llama/Llama-2-7b-chat-hf' --output {your_path}/Llama-2-7b-chat-hf-int4
    
    $ streamlit run demo/chat_streamlit.py -- --model_id {your_path}/Llama-2-7b-chat-hf-int4
