from sionna.mimo import StreamManagement
from sionna.ofdm import LMMSEEqualizer

from src.channels.cdl_channel import no
from src.ofdm.ofdm_resource_grids import rg
from src.settings.config import num_streams_per_tx, rx_tx_association


def lmmse_equalizer(response_symbols, h_est, err_var):
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)
    lmmse_equal = LMMSEEqualizer(rg, sm)

    symbols_hat, no_eff = lmmse_equal([response_symbols, h_est, err_var, no])
    return symbols_hat, no_eff


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
    from src.settings.config import batch_size, num_tx, number_of_bits
    from src.utils.plots import plot_channel_frequency_domain

    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    qam_symbols = qam_mapper(encoded_binary_values)
    mapped_qam_symbol = resource_grid_mapper(qam_symbols)
    modulated_qam_symbols = ofdm_modulation(mapped_qam_symbol)
    h_time = channel_time_domain()
    response_symbols = response_time_domain(modulated_qam_symbols, h_time)
    demodulated_symbols = ofdm_demodulation(response_symbols)

    channel_estimator = ChannelEstimator(interpolation_factor="nn")
    h_est, err_var = channel_estimator.estimate(demodulated_symbols)

    mapped_qam_symbol_hat, no_eff = lmmse_equalizer(demodulated_symbols, h_est, err_var)

    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
    print(f"QAM symbols: {qam_symbols.shape}")
    print(f"Mapped QAM symbols: {mapped_qam_symbol.shape}")
    print(f"Modulated QAM symbols: {modulated_qam_symbols.shape}")
    print(f"Response symbols: {response_symbols.shape}")
    print(f"Demodulated symbols: {demodulated_symbols.shape}")
    print(f"Estimated channel: {h_est.shape}")
    print(f"Equalized symbols: {mapped_qam_symbol_hat.shape}")

    plot_channel_frequency_domain(
        h_est[0, 0, 0, 0, 0].numpy().real.T,
    )
