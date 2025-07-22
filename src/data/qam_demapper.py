from sionna.mapping import Demapper

from src.channels.lmmse_equalizer import lmmse_equalizer


def qam_demapper(symbols_values, number_bits_per_symbol, no_eff):
    demapper = Demapper("app", "qam", number_bits_per_symbol)
    bit_values = demapper([symbols_values, no_eff])
    return bit_values


if __name__ == "__main__":
    from src.channels.cdl_channel import channel_time_domain
    from src.channels.channel_est.ls_channel import ChannelEstimator
    from src.data.binary_sources import binary_sources
    from src.data.qam_mapper import qam_mapper
    from src.data.response import response_time_domain
    from src.ldpc.ldpc_encoder import ldpc_encoder
    from src.ofdm.ofdm_demodulation import ofdm_demodulation
    from src.ofdm.ofdm_modulation import ofdm_modulation
    from src.ofdm.ofdm_resource_grids import resource_grid_mapper
    from src.settings.config import (
        batch_size,
        bits_per_symbol,
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
    response_symbols = response_time_domain(modulated_qam_symbols, h_time)
    demodulated_symbols = ofdm_demodulation(response_symbols)

    channel_estimator = ChannelEstimator(interpolation_factor="lin")
    h_est, err_var = channel_estimator.estimate(demodulated_symbols)

    mapped_qam_symbol_hat, no_eff = lmmse_equalizer(demodulated_symbols, h_est, err_var)

    binary_values_hat = qam_demapper(mapped_qam_symbol_hat, bits_per_symbol, no_eff)

    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
    print(f"QAM symbols: {qam_symbols.shape}")
    print(f"Mapped QAM symbols: {mapped_qam_symbol.shape}")
    print(f"Modulated QAM symbols: {modulated_qam_symbols.shape}")
    print(f"Response symbols: {response_symbols.shape}")
    print(f"Demodulated symbols: {demodulated_symbols.shape}")
    print(f"Estimated channel: {h_est.shape}")
    print(f"Equalized symbols: {mapped_qam_symbol_hat.shape}")
    print(f"Demapped binary values: {binary_values_hat.shape}")
