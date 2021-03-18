"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

Hier soll ein pgn datensatz extrahiert werden, der für beide finetuning modelle genutzt werden soll.
Dataset sollte so 10k - 30k Spiele beinhalten.
Fragen:
- Welche Qualität sollen die Spiele haben?
Vergleichbare Spiele, wie Noever aka Chess transformer um vergleichbarkeit zw. PGN und UCI zu haben
Brauche sowohl PGN als auch UCI notation um 2 Modelle zu trainieren und sie miteinander zu vergleichen.
- Brauche eigentlich nur UCI Daten, mit denen gpt-2 gefinetuned wird. Noever aka Chess Transformer hat ein gefinetuntes Modell für pgn

Todo:
    1. Download
    2. Convert to UCI with python chess or similar

KingBase5dataset of 2.19 million PGN
gsutil cp gs://gpt-2-poetry/data/kingbase-ftfy.txt
https://www.milibrary.org/
"""

import pickle
import chess.pgn as pgn

def read_games():
    source_file = open("data/kingbase_milibrary.PGN")  # open kingbase and milibrary in pgn
    output_file = open("data/uci_dataset.bin", 'wba')
    uci_dataset = []  # ["GAMESTRING-IN-UCI"]
    game_number = 0

    while True:
        try:
            game = pgn.read_game(source_file)
            if game is None:  # if last game
                break
            board = game.board()
            pgn_movelist = game.mainline_moves()
            uci_movelist = []
            for move in pgn_movelist:  # convert each game to uci with python chess
                uci_movelist.append(board.uci(move))
            uci_dataset.append(uci_movelist)
            game_number += 1

            if game_number % 1e4 == 0:
                print(f"{game_number} games processed")
                pickle.dump(uci_dataset, output_file)  # save so uci_dataset does not get to big
                uci_dataset = []

        except (ValueError, UnicodeDecodeError) as e:
            break

    pickle.dump(uci_dataset, output_file)
    output_file.close()

def read_dataset():
    f = open("data/uci_dataset.bin", "rb")
    dataset = pickle.load(f)
    print(len(dataset))

if __name__ == "__main__":
    read_dataset()
    #read_games()












