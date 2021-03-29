"""
Autor: Florian Babl
Thema: Probing World Knowledge of Transformer Language Models: A Case Study on Chess

PGN Dataset is being extracted and converted into UCI notation
Includes the same chess games as "Chess Tranformer" paper from the milibrary and kingbase dataset

KingBase5dataset of 2.19 million PGN gsutil cp gs://gpt-2-poetry/data/kingbase-ftfy.txt
https://www.milibrary.org/
"""

import argparse
import chess.pgn as pgn


def read_games(verbose, split_uci):
    source_file = open("data/kingbase_milibrary.PGN") # Data was combined before with shell.
    output_file = open("data/uci_dataset_with_tags.txt", 'w')
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
            # Write UCI to file, add EOS and BOS tags and [Result "1-0"] as stated in Chesstransformer paper
            output_file.write('<|startoftext|>[Result "'+game.headers["Result"] + '"] ' +
                              ' '.join(uci_movelist)+"<|endoftext|>"+"\n")  # save to file
            if verbose:
                # print('[Result "'+game.headers["Result"] + '"] ' + ' '.join(uci_movelist)+"\n", end='')
                if game_number % 1e4 == 0:  # print progress
                    print(f"{game_number} games processed")

        except (ValueError, UnicodeDecodeError) as e:  # if error
            break

    output_file.close()



from gpt_2_simple.gpt_2 import encode_plain_dataset
path_to_txt_file = "data/uci_dataset_with_tags.txt"
out_path = "data/uci_dataset_with_tags.npz"
model_name = "355M"

"""
# Method only exists in Forklab: pip install -e git+https://github.com/ForkLab/gpt-2-simple.git@dev#egg=gpt-2-simple 
Needed because file is to big to just use fine_tune
From Function description: 
Memory efficient encoder single plaint text document into compressed chunks.
For Python 3.6 only now (https://github.com/numpy/numpy/blob/1e623f8/numpy/lib/npyio.py#L745)
And need correct content inside plain text document with '<|startoftext|>' and '<|endoftext|>' in each line
"""

encode_plain_dataset(path_to_txt_file,
                     out_path=out_path,
                     model_name=model_name)

# Finetuning in Google Colab: https://colab.research.google.com/drive/1HOEoLo16CuiPe6iTSs5rV1Kf538uKRpd?usp=sharing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create UCI Dataset by Parsing PGN notation.')
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--split_uci', default=False, type=bool, help='split uci into e2 e4')
    args = parser.parse_args()
    read_games(args.verbose, args.split_uci)
