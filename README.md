# llama2.openvino
This sample shows how to implement a llama-based model with OpenVINO runtime.

<img width="947" alt="MicrosoftTeams-image (2)" src="https://github.com/OpenVINO-dev-contest/llama2.openvino/assets/91237924/c210507f-1fb2-4c68-a8d9-dae945df07d3">


- Please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints, for more [information](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- Please notice this repository is only for a functional test and personal study, and you can try to quantize the model to further optimize the performance of it

## System requirements

|                  | Description
|----------------- | ----------------------------------------
| RAM              | 128Gb + 

## Install the requirements

```
python3 -m venv openvino_env
source openvino_env/bin/activate
pip install -r requirements.txt
```

## Deployment Method 1: OpenVINO IR pipeline, export IR model from HF Optimum-Intel
**1. Run [Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference) and export the IR model**

```
python3 export_ir.py -m 'meta-llama/Llama-2-7b-hf' -o './ir_model'
cd ir_pipeline
python3 generate_op.py -m "meta-llama/Llama-2-7b-hf" -p "what is openvino?" -d "CPU"
``` 

**2. (Optional) Run restructured pipeline**:
```
    python3 generate_ir.py -m "meta-llama/Llama-2-7b-hf" -p "what is openvino?" -d "CPU"
```


## Deployment Method 2: ONNX pipeline, export ONNX model from HF Optimum
- Please notice the step below will lead to large memory consumption, you have to make sure your server should be with >256GB RAM for this step.

**1. Export the ONNX model from HuggingFace Optimum and convert it to OpenVINO IR**:

```
cd onnx_pipeline
optimum-cli export onnx --model meta-llama/Llama-2-7b-hf ./onnx_model/
mkdir ir_model
mo -m ./onnx_model/decoder_model_merged.onnx -o ./ir_model/ --compress_to_fp16
#cleanup (optional)
rm ./onnx_model/ -rf
```
**2. Run restructured pipeline**:

```
python3 generate_onnx.py -m  "meta-llama/Llama-2-7b-hf" -p "what is openvino?" -d "CPU"
```


## Interactive demo

**1. Run interactive Q&A demo with Gradio**:

```
cd demo
python3 qa_gradio.py -m "meta-llama/Llama-2-7b-hf" 
```

**2. or chatbot demo with Streamlit**:

```
python3 export_ir.py -m 'meta-llama/Llama-2-7b-chat-hf' -o './ir_model_chat'
cd demo
streamlit run chat_streamlit.py
```
