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
    2. welches format müssen daten haben
    2. Convert to UCI with python chess or similar

KingBase5dataset of 2.19 million PGN
gsutil cp gs://gpt-2-poetry/data/kingbase-ftfy.txt
https://www.milibrary.org/
"""

import argparse
import chess.pgn as pgn


def read_games(verbose, split_uci):
    source_file = open("data/kingbase_milibrary.PGN")
    output_file = open("data/uci_dataset.txt", 'w')
    game_number = 0

    while True:
        try:
            game = pgn.read_game(source_file)
            if game is None:  # if last game
                break
            board = game.board()  # create board state
            pgn_movelist = game.mainline_moves()
            uci_movelist = []
            for move in pgn_movelist:  # convert each game to uci with python chess
                if split_uci:  # Maybe used down the line to check other Encoding format
                    uci = board.uci(move)
                    uci_movelist.append(uci[:2])
                    uci_movelist.append(uci[2:])
                else:
                    uci_movelist.append(board.uci(move))
            game_number += 1
            output_file.write('[Result "'+game.headers["Result"] + '"] ' + ' '.join(uci_movelist)+"\n")  # save to file
            if verbose:
                # print('[Result "'+game.headers["Result"] + '"] ' + ' '.join(uci_movelist)+"\n", end='')
                if game_number % 1e4 == 0:  # print progress
                    print(f"{game_number} games processed")

        except (ValueError, UnicodeDecodeError) as e:  # if error
            break

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create UCI Dataset by Parsing PGN notation.')
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--split_uci', default=False, type=bool, help='split uci into e2 e4')
    args = parser.parse_args()
    read_games(args.verbose, args.split_uci)
