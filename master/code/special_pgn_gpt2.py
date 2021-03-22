"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

Hier das Modell von Philman
Uses PGN converted to readable text.

"""

from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
config = GPT2Config.from_json_file("feldman_gpt2/model/chess_model/config.json")
model = GPT2Model.from_pretrained("feldman_gpt2/model/chess_model/pytorch_model.bin", config=config)
tokenizer = GPT2Tokenizer.from_pretrained("feldman_gpt2/model/chess_model/")

if __name__ == "__main__":
    pass