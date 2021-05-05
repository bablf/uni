"""
Autor: Florian Babl (inspired by from github.com/shtoshni92)
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess


Here the Dataset for the probing classifier is created.
The resulting dataset looks like this:
{GameNumber: ("GAMESTRING-IN-PGN", "GAMESTRING-IN-UCI", RESULTING-CHESS-BOARD)}

There will be 3 settings:

# Which size should the dataset have?
    - should be arround 500k according to Chess blindfolded.
    - make sure no duplicates
    Which dataset?
     - Millionbase https://rebel13.nl/rebel13/rebel%2013.html

Feldman used 23,000 played games (converted) to finetune
Noever aka Chess Transformers use 2.1 Mil games for finetuning gpt2

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
"""

import chess
import chess.pgn as pgn
import numpy as np
import re
import json
from tqdm import tqdm


def create_dataset():
    source_file = open("data/millionbase.pgn", encoding='cp1252')  # open millionbase in pgn
    output_file = open("data/probing_dataset.jl", 'w')
    game_number = 0
    max_length = 90
    once = False
    longest = 0
    pbar = tqdm(total=5e5)

    while game_number < 5e5:  # 500k games should be enough
        try:
            game = pgn.read_game(source_file)
            if len(str(game.mainline()).split()) < 7:  # if game is too short skip it
                continue
            if game is None:  # if last game
                break
            board = game.board()
            uci_move_list = []
            board_state = []
            # Random split between 0 and 50 (40 is the reported average Movenumber per game)
            # split_game = random.randint(0, max(50, len(list(game.mainline())) - 1))

            for i, move in enumerate(game.mainline_moves()):  # convert each game to uci with python chess
                if i == max_length:  # stop at random point in pgn notation
                    break
                board_state.append(convert_board(board))
                uci_move_list.append(board.uci(move))
                board.push(move)
            board_state.append(convert_board(board))  # add last board state
            # cut part of the pgn notation. so that it is equal to uci and remove comments
            pgn_move_list = re.sub('([{]).*?([}])', "", str(game.mainline()).replace("\n", ""))
            pgn_move_list = ' '.join(pgn_move_list.split()[:135])  # 90 Moves + 45 moves numbers in pgn notation

            pgn_move_list = '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' + pgn_move_list
            uci_move_list = '<|startoftext|>[Result "' + game.headers["Result"] + '"] ' + ' '.join(uci_move_list)
            # print(uci_move_list)
            # print(pgn_move_list)
            # if board.fullmove_number*2 > max_length and once is False:
            #     max_length_uci_file = open("data/max_length_uci.txt", "w")
            #     max_length_uci_file.write(uci_move_list)
            #     max_length_uci_file.close()
            #     once = True
            # if len(pgn_move_list) > longest:
            #     max_length_pgn_file = open("data/max_length_pgn.txt", "w")
            #     max_length_pgn_file.write(pgn_move_list)
            #     max_length_pgn_file.close()
            # if pgn_move_list.count("startoftext") >= 2:
            #     print(pgn_move_list)
            #     exit()

            board_state = np.vstack(board_state)
            n = abs(max_length - board_state.shape[0])
            # if n > 0:  # Add padding to 90 with empty arrays. (used for gold standard in probing_classifier)
            #     b = np.concatenate((np.array([np.zeros(64, dtype=int)]*n), board_state), axis=0)
            # else:
            #     b = board_state
            # if b.shape != (90, 64):
            #     print("error")
            data = {"pgn": pgn_move_list, "uci": uci_move_list, "board": board_state.tolist()}
            json.dump(data, output_file)
            output_file.write('\n')
            # write everything to file. board needs to be iterated, because its too big to do str(b)
            # output_file.write(pgn_move_list + ";" + uci_move_list + ";")
            # for r in b:
            #     output_file.write(str(r)[1:-1].replace("\n", "") + " ")  # remove [ ] and newlines
            # output_file.write("\n")

            game_number += 1
            if game_number % 1e4 == 0:  # print progress
                pbar.update(1e4)
        except (ValueError, UnicodeDecodeError) as e:
            exit()

    print(game_number)
    #output_file.close()
    #open("data/numb_probing_games.txt", "w").write(str(game_number))


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
