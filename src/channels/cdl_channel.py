import tensorflow as tf
from sionna.channel import (
    ApplyOFDMChannel,
    ApplyTimeChannel,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
)
from sionna.channel.tr38901 import CDL
from sionna.utils import ebnodb2no

from src.ofdm.ofdm_resource_grids import rg
from src.settings.antenna import bs_array, ut_array
from src.settings.config import (
    bandwidth,
    batch_size,
    bits_per_symbol,
    carrier_frequency,
    cdl_model,
    code_rate,
    cyclic_prefix_length,
    delay_spread,
    direction,
    ebno_db,
    num_ofdm_symbols,
    num_subcarrier,
    speed,
    subcarrier_spacing,
)
from src.utils.plots import (
    plot_channel_frequency_complex_values,
    plot_channel_frequency_domain,
    plot_channel_tap_tim_domain,
    plot_channel_time_domain,
    plot_complex_gain,
    plot_frequency_of_each_subcarrier,
)

cdl = CDL(
    cdl_model,
    delay_spread,
    carrier_frequency,
    ut_array,
    bs_array,
    direction,
    min_speed=speed,
)


# Noises on the channel
no = ebnodb2no(ebno_db, bits_per_symbol, code_rate, rg)

# Compute the number of time steps, and the minimum and maximum time lag
l_min, l_max = time_lag_discrete_time_channel(bandwidth)
l_tot = l_max - l_min + 1
num_time_steps = num_ofdm_symbols * (num_subcarrier + cyclic_prefix_length) + l_tot - 1
num_time_samples = num_ofdm_symbols * (num_subcarrier + cyclic_prefix_length)

# Compute the channel gain and delays on time domain
complex_gain, delays = cdl(
    batch_size, num_time_steps=num_time_steps, sampling_frequency=bandwidth
)

# Compute the channel gain and delays on frequency domain
complex_gain_freq = complex_gain[
    ..., cyclic_prefix_length : -1 : (num_subcarrier + cyclic_prefix_length)
]
complex_gain_freq = complex_gain_freq[..., :num_ofdm_symbols]
frequencies = subcarrier_frequencies(num_subcarrier, subcarrier_spacing)


def channel_time_domain():
    # Channel on time domain
    h_time = cir_to_time_channel(
        bandwidth, complex_gain, delays, l_min, l_max, normalize=True
    )
    return h_time


def channel_frequency_domain():
    # Channel on frequency domain
    h_freq = cir_to_ofdm_channel(frequencies, complex_gain_freq, delays, normalize=True)
    return h_freq


def sampling_channel_freq(cdl=cdl):
    cir = cdl(batch_size, rg.num_ofdm_symbols, 1 / rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    return h_freq


def sampling_channel_time(cdl=cdl):
    cir = cdl(batch_size, num_time_steps=num_time_steps, sampling_frequency=bandwidth)
    h_time = cir_to_time_channel(bandwidth, *cir, l_min, l_max, normalize=True)
    return h_time


if __name__ == "__main__":
    plot_complex_gain(complex_gain[0, 0, 0, 0, 0], save_fig="./images/complex_gain.png")

    h_time = channel_time_domain()
    h_freq = channel_frequency_domain()

    print(f"Maximum time lag: {l_max}")
    print(f"Minimum time lag: {l_min}")
    print(f"Total time lag: {l_tot}")
    print(f"Number of time steps: {num_time_steps}")
    print(f"Complex gain: {complex_gain.shape}")
    print(f"Delays: {delays.shape}")
    print(f"Channel on time domain: {h_time.shape}")
    print(f"Channel on frequency domain: {h_freq.shape}")

    plot_channel_time_domain(
        h_time[0, 0, 0, 0, 0], save_fig="./images/channel_time_domain.png"
    )
    plot_channel_tap_tim_domain(
        h_time[0, 0, 0, 0, 0, 0],
        l_min,
        l_max,
        save_fig="./images/channel_tap_time_domain.png",
    )
    plot_frequency_of_each_subcarrier(frequencies, save_fig="./images/frequencies.png")
    plot_channel_frequency_complex_values(
        h_freq[0, 0, 0, 0, 0, 0], save_fig="./images/channel_freq_complex_values.png"
    )
    plot_channel_frequency_domain(
        h_freq[0, 0, 0, 0, 0].numpy().real.T,
        save_fig="./images/channel_freq_domain.png",
    )
