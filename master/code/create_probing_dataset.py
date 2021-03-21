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

I get the PGN Data from these sources:
1.
2.
3.

# TODO: Which size should the dataset have?
    - should be arround 50k according to Chess blindfolded.
    - make sure no duplicates
    Which dataset?
     - Millionbase https://rebel13.nl/rebel13/rebel%2013.html

Feldman used 23,000 played games (converted) to finetune
Noever aka Chess Transformers use 11k games for finetuning gpt2

"""
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.

import chess.pgn as pgn
import pickle


def create_dataset():
    source_file = open("data/MillionBase/millionbase-2.5.pgn")  # open millionbase in pgn
    output_file = open("data/pgn_uci_2_board.bin", 'wb')
    pgn_uci_2_board = []  # ("GAMESTRING-IN-PGN", "GAMESTRING-IN-UCI", RESULTING-CHESS-BOARD)
    game_number = 0
    while game_number < 1e6:  # 1 Mil games should be enough
        try:
            game = pgn.read_game(source_file)
            board = game.board()
            if game is None: # if last game
                break
            pgn_movelist = game.mainline_moves()
            print(pgn_movelist)
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
    # todo
    return board

if __name__ == "__main__":
    create_dataset()


