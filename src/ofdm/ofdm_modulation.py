from sionna.ofdm import OFDMModulator

from src.settings.config import cyclic_prefix_length


def ofdm_modulation(symbols):
    modulator = OFDMModulator(cyclic_prefix_length)
    modulated_symbols = modulator(symbols)
    return modulated_symbols


if __name__ == "__main__":
    from src.data.binary_sources import binary_sources
    from src.data.qam_mapper import qam_mapper
    from src.ldpc.ldpc_encoder import ldpc_encoder
    from src.ofdm.ofdm_resource_grids import resource_grid_mapper
    from src.settings.config import (
        batch_size,
        bits_per_symbol,
        code_rate,
        num_streams_per_tx,
        num_tx,
        number_of_bits,
    )

    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    qam_symbols = qam_mapper(encoded_binary_values)
    mapped_qam_symbol = resource_grid_mapper(qam_symbols)
    modulated_qam_symbols = ofdm_modulation(mapped_qam_symbol)

    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
    print(f"QAM symbols: {qam_symbols.shape}")
    print(f"Mapped QAM symbols: {mapped_qam_symbol.shape}")
    print(f"Modulated QAM symbols: {modulated_qam_symbols.shape}")
