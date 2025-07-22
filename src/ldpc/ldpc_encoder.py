from sionna.fec.ldpc import LDPC5GEncoder

from src.data.binary_sources import binary_sources
from src.settings.config import (
    batch_size,
    code_rate,
    num_streams_per_tx,
    num_tx,
    number_of_bits,
)


def ldpc_encoder(binary_values):
    code_words_length = int(number_of_bits / code_rate)
    encoder = LDPC5GEncoder(number_of_bits, code_words_length)

    encoded_bits = encoder(binary_values)

    return encoded_bits


if __name__ == "__main__":
    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
