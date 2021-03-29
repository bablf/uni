from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, GPT2Tokenizer
from learning_chess_blindfolded.src.data_utils.chess_tokenizer import ChessTokenizer


class PgnGPT:
    """
    This is a, on PGN notation finetuned, GPT-2 Model by Noever et. al. (Chess Transformer paper)
    with the help of gpt-2-simple.
    checkpoint downloaded from rb.gy/dsdphc
    """
    config = GPT2Config.from_json_file("noever_gpt2_checkpoint_huggingface_compatible/config.json")
    #config = GPT2Config.from_pretrained("gpt2-medium")

    # config.output_hidden_states = True
    # Don't use upper config. Has no LMHead. Other Param. are the same to gpt2
    # config.output_hidden_states = True
    # print(config)
    # model = GPT2Model.from_pretrained("pgn_checkpoint/run1/model-1000.index", from_tf=True, config=config)
    # tokenizer = GPT2Tokenizer("pgn_checkpoint/run1/encoder.json", "pgn_checkpoint/run1/vocab.bpe")
    # notation = "pgn"
    # name = "PgnGPT"


class UciGPT:
    """
    This is a, on UCI notation finetuned, GPT-2 Model with the help of gpt-2-simple.
    See file uic_gpt2.py
    """
    config = GPT2Config.from_pretrained("gpt2-medium")
    config.output_hidden_states = True
    model = GPT2Model.from_pretrained("uci_checkpoint/run1/model-18500.index", from_tf=True, config=config)
    tokenizer = GPT2Tokenizer("uci_checkpoint/run1/encoder.json", "uci_checkpoint/run1/vocab.bpe")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    name = "UciGPT"
    notation = "uci"


class PretrainedGPT:
    """
    This gpt-2 model was pretrained on UCI data by Toshniwal in his paper (Learning Chess Blindfolded).
    Model is downloaded and not further finetuned from huggingface (shtoshni/gpt2-chess-uci)
    I use his custom ChessTokenizer from his github:
    """
    config = GPT2Config.from_pretrained("shtoshni/gpt2-chess-uci")
    config.output_hidden_states = True
    model = GPT2LMHeadModel.from_pretrained("shtoshni/gpt2-chess-uci")
    vocab_file = "learning_chess_blindfolded/sample_data/lm_chess/vocab/uci/vocab.txt"
    tokenizer = ChessTokenizer(vocab_file, notation='uci', pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    notation = "uci"
    name = "PretrainedGPT"

class SpecialGPT:
    """
    This is the finetuned GPT-2 Model by Feldman from his paper.
    He finetuned on PGN notation, but converted PGN to human readable natural text with the help of a rule-based parser
    Model is not available online as of now (19.3.21). He provided his whole project (see feldman_gpt2 folder)
    """
    config = GPT2Config.from_json_file("feldman_gpt2/model/chess_model/config.json")
    config.output_hidden_states = True
    model = GPT2LMHeadModel.from_pretrained("feldman_gpt2/model/chess_model/pytorch_model.bin", config=config)
    tokenizer = GPT2Tokenizer.from_pretrained("feldman_gpt2/model/chess_model/")
    notation = "pgn"
    name = "SpecialGPT"
