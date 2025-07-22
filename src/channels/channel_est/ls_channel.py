import os

import numpy as np
import tensorflow as tf
from sionna.ofdm import LMMSEInterpolator, LSChannelEstimator

from src.channels.cdl_channel import no, sampling_channel_freq
from src.ofdm.ofdm_resource_grids import rg
from src.settings.config import (
    ebno_db,
    fft_size,
    num_bs_ant,
    num_guards,
    num_ofdm_symbols,
    num_rx,
    num_tx,
    num_ut_ant,
    speed,
)
from src.settings.ml import number_of_samples


class ChannelEstimator:
    def __init__(self, interpolation_factor: str = "nn", **kwargs):
        self.kwargs = kwargs
        self.interpolation_factor = interpolation_factor
        self.estimator = self.initial_estimatetion_methods(interpolation_factor)

    def initial_estimatetion_methods(self, interpolation_factor: str):
        if interpolation_factor.lower() == "lin":
            channel_estimator = LSChannelEstimator(rg, interpolation_type="lin")
        elif interpolation_factor.lower() == "nn":
            channel_estimator = LSChannelEstimator(rg, interpolation_type="nn")
        elif interpolation_factor.lower() == "lmmse":
            cov_mat_dir = f"data/covariance_matrices/txant_{num_ut_ant}_rxant_{num_bs_ant}_speed_{speed}_samples_{number_of_samples}_ebno_{ebno_db}"

            os.makedirs(cov_mat_dir, exist_ok=True)

            # Load covariance matrices from file
            freq_cov_mat_path = f"{cov_mat_dir}/freq_cov_mat.npy"
            time_cov_mat_path = f"{cov_mat_dir}/time_cov_mat.npy"
            space_cov_mat_path = f"{cov_mat_dir}/space_cov_mat.npy"

            # check if the files exist
            if not (
                os.path.exists(freq_cov_mat_path)
                and os.path.exists(time_cov_mat_path)
                and os.path.exists(space_cov_mat_path)
            ):
                freq_cov_mat, time_cov_mat, space_cov_mat = (
                    self.estimate_covariance_matrices(
                        num_it=self.kwargs.get("num_it", 100),
                    )
                )

                np.save(freq_cov_mat_path, freq_cov_mat.numpy())
                np.save(time_cov_mat_path, time_cov_mat.numpy())
                np.save(space_cov_mat_path, space_cov_mat.numpy())

            else:
                freq_cov_mat = tf.convert_to_tensor(
                    np.load(freq_cov_mat_path), dtype=tf.complex64
                )
                time_cov_mat = tf.convert_to_tensor(
                    np.load(time_cov_mat_path), dtype=tf.complex64
                )
                space_cov_mat = tf.convert_to_tensor(
                    np.load(space_cov_mat_path), dtype=tf.complex64
                )

            lmmse_int_time_first = LMMSEInterpolator(
                rg.pilot_pattern,
                time_cov_mat,
                freq_cov_mat,
                space_cov_mat,
                order=self.kwargs.get("order", "t-f"),
            )
            channel_estimator = LSChannelEstimator(
                rg, interpolator=lmmse_int_time_first
            )
        else:
            raise ValueError(
                f"Interpolation factor {interpolation_factor} not supported."
            )

        return channel_estimator

    def estimate(self, symbols, no=no):
        h_est, err_var = self.estimator([symbols, no])
        return h_est, err_var

    @tf.function(jit_compile=True)
    def estimate_covariance_matrices(self, num_it):
        num_rx_ant = num_bs_ant  # because Uplink so BS antennas viewed as RX antennas
        num_tx_ant = num_ut_ant  # because Uplink so UT antennas viewed as TX antennas
        num_time_steps = num_ofdm_symbols  # num_ofdm_symbols replaces num_time_steps

        # Initialize covariance matrices compatible with LMMSEInterpolator
        freq_cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)
        time_cov_mat = tf.zeros(
            [num_time_steps, num_time_steps], tf.complex64
        )  # num_time_steps replaces num_ofdm_symbols
        space_cov_mat = tf.zeros(
            [num_rx_ant, num_rx_ant], tf.complex64
        )  # RX antennas only for LMMSE

        for _ in tf.range(num_it):
            # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size] # noqa
            h_samples = sampling_channel_freq()

            #################################
            # Estimate frequency covariance
            #################################
            # Move fft_size to front: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, fft_size, num_time_steps] # noqa
            h_samples_freq = tf.transpose(h_samples, [0, 1, 2, 3, 4, 6, 5])
            # [batch_size*num_rx*num_rx_ant*num_tx*num_tx_ant, fft_size, num_time_steps]
            h_samples_freq = tf.reshape(h_samples_freq, [-1, fft_size, num_time_steps])
            # [batch_size*num_rx*num_rx_ant*num_tx*num_tx_ant, fft_size, fft_size]
            freq_cov_mat_ = tf.matmul(h_samples_freq, h_samples_freq, adjoint_b=True)
            # [fft_size, fft_size]
            freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=0)
            freq_cov_mat += freq_cov_mat_

            ################################
            # Estimate time covariance
            ################################
            # [batch_size*num_rx*num_rx_ant*num_tx*num_tx_ant, num_time_steps, fft_size]
            h_samples_time = tf.reshape(h_samples, [-1, num_time_steps, fft_size])
            # [batch_size*num_rx*num_rx_ant*num_tx*num_tx_ant, num_time_steps, num_time_steps] # noqa
            time_cov_mat_ = tf.matmul(h_samples_time, h_samples_time, adjoint_b=True)
            # [num_time_steps, num_time_steps]
            time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=0)
            time_cov_mat += time_cov_mat_

            ###############################
            # Estimate spatial covariance (RX only for LMMSE)
            ###############################
            # Move num_rx_ant to front: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size] # noqa
            h_samples_space = tf.transpose(h_samples, [0, 1, 2, 3, 4, 5, 6])
            # [batch_size*num_rx*num_tx*num_tx_ant*num_time_steps, num_rx_ant, fft_size]
            h_samples_space = tf.reshape(h_samples_space, [-1, num_rx_ant, fft_size])
            # [batch_size*num_rx*num_tx*num_tx_ant*num_time_steps, num_rx_ant, num_rx_ant] # noqa
            space_cov_mat_ = tf.matmul(h_samples_space, h_samples_space, adjoint_b=True)
            # [num_rx_ant, num_rx_ant]
            space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=0)
            space_cov_mat += space_cov_mat_

        # Normalize
        num_elements_freq = (
            num_time_steps * num_it * num_rx * num_rx_ant * num_tx * num_tx_ant
        )
        num_elements_time = (
            fft_size * num_it * num_rx * num_rx_ant * num_tx * num_tx_ant
        )
        num_elements_space = (
            fft_size * num_time_steps * num_it * num_rx * num_tx * num_tx_ant
        )

        freq_cov_mat /= tf.complex(tf.cast(num_elements_freq, tf.float32), 0.0)
        time_cov_mat /= tf.complex(tf.cast(num_elements_time, tf.float32), 0.0)
        space_cov_mat /= tf.complex(tf.cast(num_elements_space, tf.float32), 0.0)

        freq_cov_mat = self.compute_efficiency_freq_cov_mat(freq_cov_mat)
        return freq_cov_mat, time_cov_mat, space_cov_mat

    def compute_efficiency_freq_cov_mat(self, freq_cov_mat):
        # Remove row and column at dc_id
        mask = (
            tf.range(tf.shape(freq_cov_mat)[0]) != rg.dc_ind
        )  # Mask to exclude index dc_id
        freq_cov_mat = tf.boolean_mask(freq_cov_mat, mask, axis=0)  # Remove row
        freq_cov_mat = tf.boolean_mask(freq_cov_mat, mask, axis=1)  # Remove column

        # Remove rows and columns in the range [num_guards[0]:-num_guards[1]]
        freq_cov_mat = freq_cov_mat[
            num_guards[0] : -num_guards[1], num_guards[0] : -num_guards[1]
        ]
        return freq_cov_mat


if __name__ == "__main__":
    from src.channels.cdl_channel import channel_time_domain
    from src.data.binary_sources import binary_sources
    from src.data.qam_mapper import qam_mapper
    from src.data.response import response_time_domain
    from src.ldpc.ldpc_encoder import ldpc_encoder
    from src.ofdm.ofdm_demodulation import ofdm_demodulation
    from src.ofdm.ofdm_modulation import ofdm_modulation
    from src.ofdm.ofdm_resource_grids import resource_grid_mapper
    from src.settings.config import batch_size, num_streams_per_tx, number_of_bits
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

    channel_estimator = ChannelEstimator(interpolation_factor="lmmse")

    h_est, err = channel_estimator.estimate(demodulated_symbols)

    print(f"Binary values: {binary_values.shape}")
    print(f"Encoded binary values: {encoded_binary_values.shape}")
    print(f"QAM symbols: {qam_symbols.shape}")
    print(f"Mapped QAM symbols: {mapped_qam_symbol.shape}")
    print(f"Modulated QAM symbols: {modulated_qam_symbols.shape}")
    print(f"Response symbols: {response_symbols.shape}")
    print(f"Demodulated symbols: {demodulated_symbols.shape}")
    print(f"Estimated channel: {h_est.shape}")

    plot_channel_frequency_domain(
        h_est[0, 0, 0, 0, 0].numpy().real.T,
        save_fig="./images/estimated_channel_freq_domain.png",
    )

    # Get the concrete function for estimate_covariance_matrices
    # Specify the input argument num_it as a tensor
    concrete_func = (
        channel_estimator.estimate_covariance_matrices.get_concrete_function(
            num_it=tf.constant(100, dtype=tf.int32)
        )
    )

    # Extract the computational graph
    graph = concrete_func.graph

    # Set up profiling options to compute floating-point operations
    profile_options = ProfileOptionBuilder.float_operation()

    # Profile the graph to get FLOPS
    profile_result = model_analyzer.profile(graph, options=profile_options)

    # Extract the total number of floating-point operations
    total_flops = profile_result.total_float_ops

    # Print the result
    print(f"Total FLOPS for estimate_covariance_matrices: {total_flops}")

    # Optionally, compute FLOPS per second if you measure execution time
    # For example:
    import time

    start_time = time.time()
    freq_cov_mat, time_cov_mat, space_cov_mat = (
        channel_estimator.estimate_covariance_matrices(num_it=100)
    )
    execution_time = time.time() - start_time
    flops_per_second = total_flops / execution_time if execution_time > 0 else 0
    print(f"FLOPS per second: {flops_per_second}")
