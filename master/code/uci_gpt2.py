"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

Muss ich selber implementieren und finetunen.
Get uci data from create_uci_dataset
Standard tokenizer for GPT2
GPT Version: GPT2Model or GPT2LMHeadModel

"""
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import gpt_2_simple as gpt2
import os

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

file_name = "data/uci_data"
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              "data/uci_dataset.txt",
              model_name=model_name,
              steps=30000)  # steps is max number of training steps

gpt2.generate(sess, run_name='run1', checkpoint_dir='checkpoint2',model_name=model_name,
              prefix='[Result "1-0"] e2e4',  # assigns the prefix of the already played game, gets encoded automatically
              length=10)  # assigns the length of the prediction?




if __name__ == "__main__":
    pass
