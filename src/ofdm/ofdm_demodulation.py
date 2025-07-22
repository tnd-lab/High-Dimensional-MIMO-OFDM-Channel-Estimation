from sionna.ofdm import OFDMDemodulator

from src.channels.cdl_channel import l_min
from src.settings.config import cyclic_prefix_length, num_subcarrier


def ofdm_demodulation(symbols):
    demodulator = OFDMDemodulator(num_subcarrier, l_min, cyclic_prefix_length)
    demodulated_symbols = demodulator(symbols)
    return demodulated_symbols


if __name__ == "__main__":
    from src.channels.cdl_channel import sampling_channel_freq, sampling_channel_time
    from src.data.binary_sources import binary_sources
    from src.data.qam_mapper import qam_mapper
    from src.data.response import response_time_domain
    from src.ldpc.ldpc_encoder import ldpc_encoder
    from src.ofdm.ofdm_modulation import ofdm_modulation
    from src.ofdm.ofdm_resource_grids import resource_grid_mapper
    from src.settings.config import (
        batch_size,
        num_streams_per_tx,
        num_tx,
        number_of_bits,
    )
    from src.utils.plots import plot_symbols

    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    qam_symbols = qam_mapper(encoded_binary_values)
    mapped_qam_symbol = resource_grid_mapper(qam_symbols)
    modulated_qam_symbols = ofdm_modulation(mapped_qam_symbol)
    h_freq = sampling_channel_freq()
    h_time = sampling_channel_time()
    response_symbols = response_time_domain(modulated_qam_symbols, h_time)
    demodulated_symbols = ofdm_demodulation(response_symbols)
    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
    print(f"QAM symbols: {qam_symbols.shape}")
    print(f"Mapped QAM symbols: {mapped_qam_symbol.shape}")
    print(f"Modulated QAM symbols: {modulated_qam_symbols.shape}")
    print(f"Channel frequency domain: {h_freq.shape}")
    print(f"Channel time domain: {h_time.shape}")
    print(f"Response symbols: {response_symbols.shape}")
    print(f"Demodulated symbols: {demodulated_symbols.shape}")

    plot_symbols(
        demodulated_symbols[0, 0, 0].numpy().real.T,
        save_fig="./images/demodulated_symbols.png",
    )
