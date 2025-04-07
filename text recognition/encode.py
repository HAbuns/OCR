char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
max_label_len = max([len(label)for label in labels])
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def encode(label, char_to_idx, max_label_len):
    encode_labels = torch.tensor(
        [char_to_idx[char] for char in label if char in char_to_idx],
        dtype=torch.long
    )
    label_len = len(encode_labels)
    lengths = torch.tensor(
        label_len,
        dtype=torch.long
    )
    padded_labels = F.pad(
        encode_labels,
        (0, max_label_len - label_len),
        value = 0
    )
    return padded_labels, lengths