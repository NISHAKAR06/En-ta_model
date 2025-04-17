# T5 Thanglish Translator

This project fine-tunes a T5 Transformer model to translate English text into **Thanglish** (Tamil written in English script). It uses the HuggingFace Transformers and Datasets libraries.

---

## 🧠 Model Overview

The model is based on [T5 (Text-To-Text Transfer Transformer)](https://huggingface.co/docs/transformers/model_doc/t5), which treats all NLP tasks as text generation tasks. Here, we train it to convert English input into Thanglish output.

---

## 📁 Dataset

The training data is loaded from a JSON file named `en_ta_instruct.json`, where each entry follows this format:

```json
{
  "input": "Hello, how are you?",
  "output": "Vanakkam, eppadi irukkeenga?"
}
```
# Requirements:
Run the requirements file 
```
$ pip install -r requirements.txt

$ pip install transformers datasets torch
```

🏋️‍♀️ Training
To start training the model:
```
python train_t5_thanglish.py
```
Training will:

Load and preprocess the dataset

Fine-tune the t5-base model

Save checkpoints and logs

Save the final model to t5-thanglish-final/

📦 Output
After training, you will get:

t5-thanglish-final/ – the fine-tuned model and tokenizer

logs/ – training logs

🧪 Inference Example
You can load and test the model like this:
```
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-thanglish-final")
model = T5ForConditionalGeneration.from_pretrained("t5-thanglish-final")

text = "translate English to Thanglish: I am going to school"
input_ids = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=128)

print("Thanglish:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

⚙️ Notes
If you encounter an error like unexpected keyword argument 'evaluation_strategy', your transformers library may be outdated. Upgrade it using:
```
$ pip install --upgrade transformers
```
GPU support via CUDA will speed up training. Ensure your PyTorch installation supports CUDA if available.
