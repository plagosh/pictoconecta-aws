from itertools import chain
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

def build_input_from_segments(history, reply, tokenizer, with_eos=True):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    
    # Concatenar la personalidad, el historial y la respuesta del usuario
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    
    # Asignar token_type_ids para toda la entrada del usuario y para la salida de GPT-2
    token_type_ids = [speaker1] * sum(len(s) for s in sequence[:-1]) + [speaker2] * len(sequence[-1])
    
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = token_type_ids
    return instance

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    if num_added_tokens > 0:
        model.resize_token_embeddings(orig_num_tokens + num_added_tokens)
