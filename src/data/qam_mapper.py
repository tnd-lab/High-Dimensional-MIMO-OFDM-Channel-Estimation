from sionna.mapping import Mapper

from src.data.binary_sources import binary_sources
from src.ldpc.ldpc_encoder import ldpc_encoder
from src.settings.config import (
    batch_size,
    bits_per_symbol,
    num_streams_per_tx,
    num_tx,
    number_of_bits,
)


def qam_mapper(binary_values):
    mapper = Mapper("qam", bits_per_symbol)
    mapped_values = mapper(binary_values)
    return mapped_values


if __name__ == "__main__":
    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    qam_symbols = qam_mapper(encoded_binary_values)
    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
    print(f"QAM symbols: {qam_symbols.shape}")
