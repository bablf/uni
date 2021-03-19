from transformers import GPT2Config,GPT2LMHeadModel, GPT2Model, GPT2Tokenizer
from learning_chess_blindfolded.src.data_utils.chess_tokenizer import ChessTokenizer


class PgnGPT:
    """
    This is a, on PGN notation finetuned, GPT-2 Model by Noever et. al. (Chess Transformer paper)
    with the help of gpt-2-simple.
    checkpoint downloaded from rb.gy/dsdphc
    """
    config = GPT2Config.from_json_file("noever_gpt2_checkpoint_huggingface_compatible/config.json")
    model = GPT2Model.from_pretrained("checkpoint/run1/model-1000.index", from_tf=True, config=config)
    tokenizer = GPT2Tokenizer("checkpoint/run1/encoder.json", "checkpoint/run1/vocab.bpe")
    #config.output_hidden_states = True


class UciGPT:
    """
    Todo : to be finetuned
    This is a, on UCI notation finetuned, GPT-2 Model with the help of gpt-2-simple.
    See file uic_gpt2.py
    """
    config = GPT2Config.from_json_file("babl_gpt2_checkpoint_huggingface_compatible/config.json")
    model = GPT2Model.from_pretrained("checkpoint2/run1/model-1000.index", from_tf=True, config=config)
    tokenizer = GPT2Tokenizer("checkpoint2/run1/encoder.json", "checkpoint2/run1/vocab.bpe")


class PretrainedGPT:
    """
    This gpt-2 model was pretrained on UCI data by Toshniwal in his paper (Learning Chess Blindfolded).
    Model is downloaded and not further finetuned from huggingface (shtoshni/gpt2-chess-uci)
    I use his custom ChessTokenizer from his github:
    """
    model = GPT2LMHeadModel.from_pretrained("shtoshni/gpt2-chess-uci")
    vocab_file = "learning_chess_blindfolded/sample_data/lm_chess/vocab/uci/vocab.txt"
    tokenizer = ChessTokenizer(vocab_file, notation='uci', pad_token="<pad>", bos_token="<s>", eos_token="</s>")

class SpecialGPT:
    """
    This is the finetuned GPT-2 Model by Feldman from his paper.
    He finetuned on PGN notation, but converted PGN to human readable natural text with the help of a rule-based parser
    Model is not available online as of now (19.3.21). He provided his whole project (see feldman_gpt2 folder)
    """
    config = GPT2Config.from_json_file("feldman_gpt2/model/chess_model/config.json")
    model = GPT2LMHeadModel.from_pretrained("feldman_gpt2/model/chess_model/pytorch_model.bin", config=config)
    tokenizer = GPT2Tokenizer.from_pretrained("feldman_gpt2/model/chess_model/")
