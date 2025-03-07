# SimpleAITestbed
Most basic testbed for streaming responses from Ollama or OpenAI and prompts from file.

## Installation
Install requirements.txt using pip:
> pip install -r requirements.txt

Otherwise, install requirements manually in your favorite IDE.

## Usage
Add your prompt to prompt.txt or leave prompt.txt empty and add prompt manually

### Setting the context size

The OpenAI API does not have a way of setting the context size for a model. If you need to change the context size, create a `Modelfile` which looks like:

```
FROM <some model>
PARAMETER num_ctx <context size>
```

Use the `ollama create mymodel` command to create a new model with the updated context size. Call the API with the updated model name.



## Example Running
Features Streaming tokens. Prior to first token:
<img width="803" alt="image" src="https://github.com/user-attachments/assets/c6e0f7e3-6c14-4b24-a9c9-5de1067498c8" />


While streaming and responding:
<img width="940" alt="image" src="https://github.com/user-attachments/assets/26e37054-9e73-46c8-a566-f7011e498227" />
