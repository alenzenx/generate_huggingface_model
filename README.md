# This repo can teach you how to edit and custom Hugging Face format LLM model

## Quick Start

#### *Notice : Please install in order*

## Setting up a virtual environment
```
python -m venv "your path"
```

## Activate virtual environment
```
"your path"\Scripts\Activate.ps1
```

## Install huggingface edit version
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
```
pip install -r requirements.txt
```
```
git clone https://github.com/huggingface/transformers.git
```
```
cd transformers
```
```
pip install -e .
```

## Officially recommended "modular calling" large language model local editing solution : use modular_*.py and configuration_*.py generate modeling_*.py (this is qwen official editing method)


#### (In this example)
#### Put folder "qwen_light" in "transformers/src/transformers/models"
#### Put file "auto/configuration_auto.py" in transformers/src/transformers/models/auto replace original file

## cd in transformers
```
cd transformers
```

## Run modular_model_converter
```
python utils/modular_model_converter.py --files_to_parse src/transformers/models/qwen_light/modular_qwenlight.py
```

## Inference
```
python inference.py
```