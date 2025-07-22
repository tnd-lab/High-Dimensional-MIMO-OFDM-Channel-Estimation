import os

import numpy as np
import tensorflow as tf
from sionna.ofdm import RemoveNulledSubcarriers
from tqdm import tqdm

from src.channels.cdl_channel import sampling_channel_freq
from src.data.binary_sources import binary_sources
from src.data.qam_mapper import qam_mapper
from src.data.response import response_freqency_domain
from src.ldpc.ldpc_encoder import ldpc_encoder
from src.ofdm.ofdm_resource_grids import resource_grid_mapper, rg
from src.settings.config import (
    batch_size,
    ebno_db,
    num_bs_ant,
    num_effective_subcarriers,
    num_ofdm_symbols,
    num_streams_per_tx,
    num_tx,
    num_ut_ant,
    number_of_bits,
    pilots,
    speed,
)
from src.settings.ml import number_of_samples

remove_nulls_subcarriers = RemoveNulledSubcarriers(rg)


def flatten_last_dims(tensor, num_dims=2):
    if num_dims == len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tf.shape(tensor)
        last_dim = tf.reduce_prod(tensor.shape[-num_dims:])
        new_shape = tf.concat([shape[:-num_dims], [last_dim]], 0)

    return tf.reshape(tensor, new_shape)


def get_mask(pilot_ind):
    mask = np.zeros(
        (num_tx, num_streams_per_tx, num_ofdm_symbols * num_effective_subcarriers)
    )
    num_pilots_per_stream = int(
        num_effective_subcarriers * len(pilots) / (num_tx * num_streams_per_tx)
    )
    for i_tx in range(num_tx):
        for i_stream in range(num_streams_per_tx):
            pilot_positions = np.array(
                [
                    i_tx * num_streams_per_tx
                    + i_stream
                    + k * num_tx * num_streams_per_tx
                    for k in range(num_pilots_per_stream)
                ]
            )
            mask[i_tx, i_stream, pilot_ind[pilot_positions]] = 1

    mask = mask.reshape(
        num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers
    )
    return mask


def get_pilot_matrix(response_matrix):
    removed_response_matrix = remove_nulls_subcarriers(response_matrix).numpy()

    removed_response_matrix = np.repeat(
        removed_response_matrix[:, :, :, np.newaxis, np.newaxis, :, :],
        num_streams_per_tx,
        axis=-3,
    )

    # get pilot indices
    pilot_ind = tf.argsort(
        flatten_last_dims(rg.pilot_pattern.mask), axis=-1, direction="DESCENDING"
    )
    pilot_ind = pilot_ind[0, 0, : rg.num_pilot_symbols]

    # get mask from pilot indices
    mask = get_mask(pilot_ind.numpy())

    # apply mask
    new_response_matrix = (
        removed_response_matrix * mask[np.newaxis, np.newaxis, np.newaxis, ...]
    )

    return new_response_matrix


def synthesys_training_data():
    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    qam_symbols = qam_mapper(encoded_binary_values)
    mapped_qam_symbol = resource_grid_mapper(qam_symbols)
    h_freq = sampling_channel_freq()
    response_matrix = response_freqency_domain(mapped_qam_symbol, h_freq)

    # remove guard bands and null subcarriers
    h_freq = remove_nulls_subcarriers(h_freq)

    return response_matrix, h_freq


if __name__ == "__main__":
    src_dir = f"txant_{num_ut_ant}_rxant_{num_bs_ant}_speed_{speed}_samples_{number_of_samples}_ebno_{ebno_db}"

    h_freqs = []
    pilot_matrices = []
    for i in tqdm(range(number_of_samples)):
        response_matrix, h_freq = synthesys_training_data()
        h_freq = h_freq.numpy()
        pilot_matrix = get_pilot_matrix(response_matrix)

        h_freqs.append(h_freq)
        pilot_matrices.append(pilot_matrix)

    # save the data
    os.makedirs(f"data/{src_dir}", exist_ok=True)

    np.save(f"data/{src_dir}/h_freqs.npy", h_freqs)
    np.save(f"data/{src_dir}/pilot_matrices.npy", pilot_matrices)
