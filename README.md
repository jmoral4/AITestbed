# SimpleAITestbed
Most basic testbed for streaming responses from Ollama or OpenAI and prompts from file.

## Installation via Pycharm
* Open Pycharm, Open and point to SimpleAITestBed
* Pycharm should offer to install dependencies from Requirements.txt and setup venv
* Select main.py and  Click run after project fully loads.

## Installation (manual)
Install requirements.txt using pip:
> pip install -r requirements.txt

Otherwise, install requirements manually in your favorite IDE using pip or IDE methods

## Usage
Add your prompt to prompt.txt or leave prompt.txt empty and add prompt manually

## Setting the context size

The OpenAI API does not have a way of setting the context size for a model. If you need to change the context size, create a `Modelfile` which looks like:

```
FROM <some model>
PARAMETER num_ctx <context size>
```

Use the `ollama create mymodel` command to create a new model with the updated context size. Call the API with the updated model name.

# Features :D
* Streaming responses
* Configurable System and User Prompt from file


## Example Running
Outputs model info for debug purposes. Can be commented out.
<img width="996" alt="image" src="https://github.com/user-attachments/assets/91a1a463-0800-49b0-ba75-3d5fb6661fdb" />

