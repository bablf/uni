import json
from tqdm import tqdm

with open("data/probing_dataset.jl", encoding='cp1252') as f:
    longest = 0
    y = set()
    for line in tqdm(f):
        r = json.loads(line)
        x = [move[-1] for move in r["pgn"].split() if "." not in move and "Result" not in move]
        y.update(x)
    print(y)

#{']', '4', '7', '#', 'B', 'N', 'O', '3', '+', '1', 'R', '8', 'Q', '6', '2', '5'}
#'b4', 'h2', 'e3', '8#', '8+', 'f4', 'f1', 'b7', 'd5', 'R#', '-O', 'O#', 'c7', 'c8', 'b1', 'b2', 'h4', '=B', 'd1', '6+', 'c3', 'g5', 'a1', '1+', '4+', 'N#', 'e2', 'a7', 'a4', '=Q', 'g4', '6#', 'd7', 'a5', 'f3', 'c6', 'g6', 'N+', 'c4', '1#', '"]', 'd3', 'b5', 'g1', 'd8', 'B+', 'c2', 'f8', 'h5', 'f6', 'O+', 'Q+', 'Q#', 'h8', 'd2', 'd6', '2#', 'f7', 'h1', 'e7', 'h6', '=R', 'c5', 'd4', 'g3', 'b6', 'b8', '2+', 'g7', 'f5', 'B#', '4#', 'h7', 'e1', 'R+', 'b3', '=N', 'e4', 'a2', 'e6', 'a6', 'h3', 'g8', 'e5', '5+', '3+', 'c1', '5#', 'a3', '7+', 'g2', '3#', '7#', 'f2', 'a8', 'e8'