from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel

from src.channels.cdl_channel import l_tot, no, num_time_samples


def response_time_domain(time_symbols, h_time, no=no):
    # Apply the time domain channel response
    channel_time = ApplyTimeChannel(
        num_time_samples=num_time_samples, l_tot=l_tot, add_awgn=True
    )
    time_response_symbols = channel_time([time_symbols, h_time, no])
    return time_response_symbols


def response_freqency_domain(freq_symbols, h_freq, no=no):
    # Apply the frequency domain channel response
    channel_freq = ApplyOFDMChannel(add_awgn=True)
    freq_response_symbols = channel_freq([freq_symbols, h_freq, no])
    return freq_response_symbols
