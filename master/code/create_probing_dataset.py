"""
Autor: Florian Babl (inspired by from github.com/shtoshni92)
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess


Here the Dataset for the probing classifier is created.
The resulting dataset looks like this:
{GameNumber: ("GAMESTRING-IN-PGN", "GAMESTRING-IN-UCI", RESULTING-CHESS-BOARD)}

There will be 3 settings:

# TODO: setting mit beendetem Spiel
# TODO: setting mitten im Spiel
# TODO: setting am Anfang des Spiels

# TODO: Which size should the dataset have?
    - should be arround 50k according to Chess blindfolded.
    - make sure no duplicates
    Which dataset?
     - Millionbase https://rebel13.nl/rebel13/rebel%2013.html

Feldman used 23,000 played games (converted) to finetune
Noever aka Chess Transformers use 2.1 Mil games for finetuning gpt2

"""
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.

"""
Format of dataset
    Option 1: 
#       - PGN, UCI, Gamedevelopment on board
#       --> Gamedeveleopment on board: 8x8xGameLenght
#       --> gameLenght varies from game to game. Is that a problem? 
        for move in pgn/uci/special:
            actual board state = calculate_board_state
            hidden states = give list of moves till now to finetuned model 
            prediction for board = probing(hidden states)
            compare with actual board state. 
#   =====> Idee verworfen, weil könnte schwer werden zu implementieren. 
    # TODO: Option 2: 
            Cut PGN/UCI at some point (1/3 opening, 1/3 mid game, 1/3 late game/cut)
"""

import chess.pgn as pgn
import pickle
import chess
import numpy as np
import random
import math
from tqdm import tqdm
import re
import string


def create_dataset():
    source_file = open("data/2005.pgn", encoding='cp1252')  # open millionbase in pgn # todo change to millionbase
    output_file = open("data/probing_dataset.txt", 'w')
    game_number = 0
    max_length = 0
    pbar = tqdm(total=5e5)
    while game_number < 5e5:  # 500k games should be enough
        try:
            game = pgn.read_game(source_file)
            if game is None:  # if last game
                break
            board = game.board()
            uci_move_list = []
            split_game = random.randint(0, len(list(game.mainline())) - 1)

            for i, move in enumerate(game.mainline_moves()):  # convert each game to uci with python chess
                if i == split_game:  # stop at random point in pgn notation
                    break
                uci_move_list.append(board.uci(move))
                board.push(move)
            # cut part of the pgn notation. so that it is equal to uci
            split_game += math.ceil(split_game / 2)
            pgn_move_list = re.sub("([{]).*?([}])", "", str(game.mainline()).replace("\n", ""))
            pgn_move_list = ' '.join(pgn_move_list.split()[:split_game])
            pgn_move_list = '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' + pgn_move_list
            uci_move_list = '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' + ' '.join(uci_move_list)

            if split_game > max_length:
                max_length = split_game
                max_length_pgn_file = open("data/max_length_pgn.txt", "w")
                max_length_uci_file = open("data/max_length_uci.txt", "w")
                max_length_uci_file.write('<|startoftext|>[Result "' + game.headers["Result"] + '"] ' +
                                          ' '.join(uci_move_list))
                max_length_pgn_file.write(pgn_move_list)
                max_length_pgn_file.close()
                max_length_uci_file.close()

            board = convert_board(board)  # convert board to format I need
            # print(pgn_move_list + ";" + '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' +
            #                    ' '.join(uci_move_list) + ";" + str(board).replace("\n", "")[1:-1] + "\n")
            output_file.write(pgn_move_list + ";" + uci_move_list + ";" + str(board[1:-1]).replace("\n", "") + "\n")
            # pgn_uci_2_board.append((pgn_move_list, '<|startoftext|>[Result "'+game.headers["Result"] + '"] ' +
            #                         ' '.join(uci_move_list), board))
            game_number += 1

            if game_number % 1e4 == 0:  # print progress
                pbar.update(1e4)
        except (ValueError, UnicodeDecodeError) as e:
            pass
    print(game_number)
    #max_length_file.write(str(max_length))
    #pickle.dump(pgn_uci_2_board, output_file)
    output_file.close()
    open("data/numb_probing_games.txt","w").write(str(game_number))


def convert_board(board):
    """
    Taken from python-chess/issues/404
    Converts the board to squared_list (see wikipedia)
    """
    l = [0] * 64
    for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):
        l[sq] = board.piece_type_at(sq)
    for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):
        l[sq] = -board.piece_type_at(sq)
    # print(np.array(l).reshape(8, 8))
    return np.array(l).reshape(64)


if __name__ == "__main__":
    create_dataset()
