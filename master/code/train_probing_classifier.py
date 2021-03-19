"""
Autor: Florian Babl
Thema: Probing Classifiers for LM in a Chess setting
"""


""" GPT-2 MODELLE """
# Auf welche Modelle habe ich zugriff?
#   1. Auf gpt2 mit UCI & PGN notation finetunen (Muss ich selber trainieren/programmieren)
#       habe PGN Model from Noever aka Chess Transformer
#       Muss UCI Model implementieren
#   2. Auf gpt2 mit converted PGN finetunen
#   3. pretrained gpt2 (kann einfach aus Huggingface importiert werden) # PGN to UCI exists. From Blindfolded Paper
# Todo: Alle 4 Modelle implementieren und aufrufen können.
# Todo selbst implementiertes finetuning trainieren lassen.
# Todo: brauche Daten um finetuning zu machen.



""" PROBING """
# Todo: Wie kann Hidden State extrahiert werden
# TODO: Probing Classifier erstellen, und trainieren.
# 1. Neuronales Netz ausdenken für alle Probing Classifier
# 2. Few-short results implementieren für gpt-2 # Todo: in few-shot einlesen. in bezug auf fine-tuning (mit oder ohne)
# Todo: Datensatz für probing erstellen (Muss der der gleiche für beide finetuned GPT2 modelle sein? Wäre gut.)
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.

import argparse
from create_probing_dataset import create_dataset
from models import PretrainedGPT, PgnGPT, UciGPT, SpecialGPT
# create or open probing_dataset
if not "data/probing_dataset":
    create_dataset()
probing_data = open("data/probing_dataset")

models = [PretrainedGPT, PgnGPT, UciGPT, SpecialGPT]

def train_probing_classifier(model, ):

    # encode all trainingdata
    # for all training_data:
    #


# for 4 models
for model in models: # ["pretrained_gpt","pgn_gpt", "uci_gpt", "special_gpt"]
    train_probing_classifier(model)
    save_to
#   give some chessgame prefix and let it generate a move. #Todo: (all, opening, middle or late game -task)
#   only care about hidden state that determined move
#   train probing classifier with chess board from dataset, so we can tell why the model made the move
#   evaluate probing classifier
#   save to file
# compare results

parser = argparse.ArgumentParser(
    description="Easily retrain OpenAI's GPT-2 text-generating model on new texts. (https://github.com/minimaxir/gpt-2-simple)"
)

# Explicit arguments
parser.add_argument(
    '--train_models', help='choose which models to train. Select from: [PretrainedGPT, PgnGPT, UciGPT, SpecialGPT].'
                           'Default is all of them.',default=[PretrainedGPT, PgnGPT, UciGPT, SpecialGPT], type=list)
parser.add_argument(
    '--run_name', help="[finetune/generate] Run number to save/load the model",
    nargs='?', default='run1')
parser.add_argument(
    '--checkpoint_dir', help="[finetune] Path of the checkpoint directory",
    nargs='?', default='checkpoint')
if __name__ == '__main__':
    pass


