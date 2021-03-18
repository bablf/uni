import tensorflow as tf
# pip install git+https://github.com/huggingface/transformers.git
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

text_list = [' resigns. ', 'wins. ']

# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# to put the model in a named directory, load the model and tokenizer and then save (as per https://huggingface.co/transformers/quickstart.html):
#tokenizer.save_pretrained("./gpt2")
#model.save_pretrained("./gpt2")

# options are (from https://huggingface.co/transformers/pretrained_models.html)
# 'gpt2' : 12-layer, 768-hidden, 12-heads, 117M parameters. # OpenAI GPT-2 English model
# 'gpt2-medium' : 24-layer, 1024-hidden, 16-heads, 345M parameters. # OpenAI’s Medium-sized GPT-2 English model
# 'gpt2-large' : 36-layer, 1280-hidden, 20-heads, 774M parameters. # OpenAI’s Large-sized GPT-2 English model
# 'gpt2-xl' : 48-layer, 1600-hidden, 25-heads, 1558M parameters.. # OpenAI’s XL-sized GPT-2 English model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
#model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model = TFGPT2LMHeadModel.from_pretrained("../data/chess_model", pad_token_id=tokenizer.eos_token_id, from_pt=True)

for text in text_list:
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(text, return_tensors='tf')

    # generate text until the output length (which includes the context length) reaches 50
    output_list  = model.generate(
        input_ids,
        do_sample=True,
        max_length=200,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3)

    print("\n{}:".format(text))
    for i, beam_output  in enumerate(output_list):
        output = tokenizer.decode(beam_output , skip_special_tokens=True)
        print("\t[{}]: {}".format(i, " ".join(output.split())))
