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
#   =====> Idee verworfen, weil könnte schwer werden zu implementieren. 
        for move in pgn/uci/special:
            actual board state = calculate_board_state
            hidden states = give list of moves till now to finetuned model 
            prediction for board = probing(hidden states)
            compare with actual board state. 
    Option 2: 
        Cut PGN/UCI at some point (1/3 opening, 1/3 mid game, 1/3 late game/cut)
"""


import chess.pgn as pgn
import pickle
import chess
import numpy as np

def create_dataset():
    source_file = open("data/millionbase.pgn", encoding='cp1252')  # open millionbase in pgn
    output_file = open("data/pgn_uci_2_board.txt", 'w')
    pgn_uci_2_board = []  # ("GAMESTRING-IN-PGN", "GAMESTRING-IN-UCI", RESULTING-CHESS-BOARD)
    game_number = 0
    lines = ""

    while game_number < 1e6:  # 1 Mil games should be enough
        try:
            game = pgn.read_game(source_file)
            board = game.board()
            if game is None: # if last game
                break
            pgn_movelist = game.mainline_moves()
            print("test ", pgn_movelist)
            uci_movelist = []
            for move in pgn_movelist:  # convert each game to uci with python chess

                uci_movelist.append(board.uci(move))
                board.push(move)
            board = convert_board(board)  # Todo: convert board to format I need
            pgn_uci_2_board[game_number] = (str(pgn_movelist), ' '.join(uci_movelist), board)
            game_number += 1
        except (ValueError, UnicodeDecodeError) as e:
            pass

    pickle.dump(pgn_uci_2_board, output_file)
    output_file.close()

def convert_board(board):
    """Taken from python-chess/issues/404"""
    l = [0] * 64
    print(board)
    for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):
        l[sq] = board.piece_type_at(sq)
    print(np.array(l).reshape(8, 8))

    for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):
        l[sq] = -board.piece_type_at(sq)
    # print(np.array([board.piece_type_at(sq) for sq in chess.SQUARES]).reshape(8,8))
    print(np.array(l).reshape(8, 8))
    return board

if __name__ == "__main__":
    create_dataset()
    # '{ORS = sub(/[^\]]$/,"") ? "" : "\n"} 1'
    #

