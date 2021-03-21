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
# Todo selbst implementiertes finetuning trainieren lassen.
# Todo: brauche Daten um finetuning zu machen.



""" PROBING """
# Todo: Wie kann Hidden State extrahiert werden
# TODO: Probing Classifier erstellen, und trainieren.
# 1. Neuronales Netz ausdenken für alle Probing Classifier
# 2. Few-short results implementieren für gpt-2 # Todo: in few-shot einlesen. in bezug auf fine-tuning (mit oder ohne)
# Todo: Datensatz für probing erstellen
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.

import argparse
from create_probing_dataset import create_dataset
from models import PretrainedGPT, PgnGPT, UciGPT, SpecialGPT
from torch.utils.data import DataLoader, TensorDataset, random_split
from probing_classifier import ProbingChess
# create or open probing_dataset
if not "data/probing_dataset":
    create_dataset()
probing_data = open("data/probing_dataset")

models = [PretrainedGPT, PgnGPT, UciGPT, SpecialGPT]
def encode_game(model, game):
    # Todo what do tokenizer return?
    if model.notation == "pgn":  # Todo for specialGPT
        return model.tokenizer(game[0])
    else:  # uci
        return model.tokenizer(game[1])

def train_probing_classifier(model, train_data):
    for game in train_data:
        enc = encode_game(model, game)  # todo batchsize =16 ?
        pt_outputs = model(**enc)  # todo check  shape (batch_size, sequence_length, hidden_size)
        ProbingChess(pt_outputs.hidden_states)
    # encode all trainingdata
    # for all training_data:
    #


# for 4 models

for model in models: # ["pretrained_gpt","pgn_gpt", "uci_gpt", "special_gpt"]
    train_probing_classifier(model)
    # todo save
#   give some chessgame prefix and let it generate a move. #Todo: train on all but test on beginning, middel, lategame
#   only care about hidden state that determined move
#   train probing classifier with chess board from dataset, so we can tell why the model made the move
#   evaluate probing classifier
#   save to file
# compare results

# todo preprocessing training data
batch_size = 512 # todo check if I can store output from models somehow and load them to the Dataloader
train_loader = DataLoader(train, batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size)
test_loader = DataLoader(test_dataset, batch_size)

def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss": loss.item(), "accuracy": accuracy.item()}

def fit(model, train_loader, val_loader, epochs, lr, optimizer_function=torch.optim.SGD):
    history = []
    optimizer = optimizer_function(model.parameters(), lr)
    for epoch in range(epochs):
        print("Epoch ", epoch)
        # Train
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validate
        for batch in val_loader:
            result = evaluate(model, val_loader)
        print("loss: ", result["loss"], "accuracy: ", result["accuracy"], "\n")
        history.append(result)

    return history

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


