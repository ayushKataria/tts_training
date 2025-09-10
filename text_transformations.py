import torch
from constants import hp


def text_norm(text):
    """
    Define the text normalization operations that need to be done on the text before training.
    We are using LJ speech where this is already done like opening abbreviations, converting numbers in words etc.
    This is a dummy function which only does lowercase and removes whitespaces, but a more meaningful one can be described if needed

    Args:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text.
    """
    return text.lower().strip()


symbol_to_id = {s: i for i, s in enumerate(hp.symbols)}


def text_to_seq(text):
    text = text.lower()
    seq = []

    for s in text:
        _id = symbol_to_id.get(s, None)
        if _id is not None:
            seq.append(_id)

    seq.append(symbol_to_id["EOS"])

    return torch.IntTensor(seq)


if __name__ == "__main__":
    print(text_to_seq("Hello World"))