def decode(encode_sequences, idx_to_char, blank_char='-'):
    decode_sequences = []

    for seq in encode_sequences:
        decode_label = []
        for idx, token in enumerate(seq):
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decode_label.append(char)

        decode_sequences.append(''.join(decode_label))

    return decode_sequences