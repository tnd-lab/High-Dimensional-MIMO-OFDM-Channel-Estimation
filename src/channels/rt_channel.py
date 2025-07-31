import tensorflow as tf
import math
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.channel import (
    cir_to_ofdm_channel,
    cir_to_time_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    time_to_ofdm_channel,
)
from sionna.utils import ebnodb2no
from src.settings.config import (
    bandwidth,
    cyclic_prefix_length,
    num_ofdm_symbols,
    num_subcarrier,
    speed,
    subcarrier_spacing,
    num_ut_ant,
    num_bs_ant,
    ebno_db,
    bits_per_symbol,
    code_rate,
)
from src.ofdm.ofdm_resource_grids import rg
import random
import numpy as np

# Noises on the channel
no = ebnodb2no(ebno_db, bits_per_symbol, code_rate, rg)
l_min, l_max = time_lag_discrete_time_channel(bandwidth)
l_tot = l_max - l_min + 1
num_time_steps = num_ofdm_symbols * (num_subcarrier + cyclic_prefix_length) + l_tot - 1

default_scene_template = "/Users/rysheng/Blender/maps/polymtl_1/polymtl.xml"
default_ut_loc = [-28.7637, -79.456, 1.359615]
default_bs_loc = [33.0255, 26.4153, 13.7488]
default_ut_angle = [math.pi / 3, -math.pi / 9, 0]
default_bs_angle = [-math.pi / 1.5, 0, 0]


def add_random_offset(location, max_offsets):
    return [
        coord + random.uniform(-delta, delta)
        for coord, delta in zip(location, max_offsets)
    ]


def setup_scenario(
    scene_template: str = default_scene_template,
    ut_loc: list = default_ut_loc,
    bs_loc: list = default_bs_loc,
    ut_angle: list = default_ut_angle,
    bs_angle: list = default_bs_angle,
):
    scene = load_scene(scene_template)

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1 if num_ut_ant == 1 else int(num_ut_ant // 2),
        polarization_model=1 if num_ut_ant == 1 else 2,
        polarization="V" if num_ut_ant == 1 else "cross",
        pattern="tr38901",
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1 if num_bs_ant == 1 else int(num_bs_ant // 2),
        polarization_model=1 if num_bs_ant == 1 else 2,
        polarization="V" if num_ut_ant == 1 else "cross",
        pattern="tr38901",
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
    )

    tx = Transmitter("tx", ut_loc, ut_angle)
    scene.add(tx)

    rx = Receiver("rx", bs_loc, bs_angle)
    scene.add(rx)

    return scene


def filter_gain(complex_gain, delay):
    mask = tf.squeeze(delay != -1.0, axis=[0, 1, 2])  # shape: [27]

    tau_filtered = tf.boolean_mask(delay, mask, axis=3)

    a_filtered = tf.boolean_mask(complex_gain, mask, axis=5)
    return a_filtered, tau_filtered


def rt_channel_freq(
    scene_template: str = default_scene_template,
    ut_loc: list = default_ut_loc,
    bs_loc: list = default_bs_loc,
    ut_angle: list = default_ut_angle,
    bs_angle: list = default_bs_angle,
):
    scene = setup_scenario(
        scene_template=scene_template,
        ut_angle=ut_angle,
        bs_angle=bs_angle,
        ut_loc=ut_loc,
        bs_loc=bs_loc,
    )
    paths = scene.compute_paths(los=False)
    
    paths.apply_doppler(
        sampling_frequency=subcarrier_spacing,
        num_time_steps=num_ofdm_symbols,
        tx_velocities=[
            speed / math.sqrt(2),
            speed / math.sqrt(2),
            0,
        ],
        rx_velocities=[0, 0, 0],
    )  # Or rx speeds
    a, tau = paths.cir(los=False)

    mask = tf.squeeze(tau > 0, axis=[0, 1, 2])
    tau = tf.boolean_mask(tau, mask, axis=3)
    a = tf.boolean_mask(a, mask, axis=5)
    
    frequencies = subcarrier_frequencies(num_subcarrier, subcarrier_spacing)

    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    return h_freq


def rt_channel_time(
    scene_template: str = default_scene_template,
    ut_loc: list = default_ut_loc,
    bs_loc: list = default_bs_loc,
    ut_angle: list = default_ut_angle,
    bs_angle: list = default_bs_angle,
):
    scene = setup_scenario(
        scene_template=scene_template,
        ut_angle=ut_angle,
        bs_angle=bs_angle,
        ut_loc=ut_loc,
        bs_loc=bs_loc,
    )
    paths = scene.compute_paths(los=False)
    paths.apply_doppler(
        sampling_frequency=bandwidth,
        num_time_steps=num_time_steps,
        tx_velocities=[
            speed / math.sqrt(2),
            speed / math.sqrt(2),
            0,
        ],
        rx_velocities=[0, 0, 0],
    )  # Or rx speeds
    a, tau = paths.cir(los=False)
    
    mask = tf.squeeze(tau > 0, axis=[0, 1, 2])
    tau = tf.boolean_mask(tau, mask, axis=3)
    a = tf.boolean_mask(a, mask, axis=5)

    h_time = cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=True)
    return h_time


def sampling_rt_channel_freq():
    max_offset_loc_ut = [1.0, 1.0, 1.0]
    max_offset_angle_ut = [math.pi / 12, math.pi / 12, math.pi / 12]
    random_ut_loc = add_random_offset(default_ut_loc, max_offset_loc_ut)
    random_ut_angle = add_random_offset(default_ut_angle, max_offset_angle_ut)

    h_freq = rt_channel_freq(
        ut_loc=random_ut_loc,
        ut_angle=random_ut_angle,
    )
    return h_freq


def sampling_rt_channel_time():
    max_offset_loc_ut = [1.0, 1.0, 1.0]
    max_offset_angle_ut = [math.pi / 12, math.pi / 12, math.pi / 12]
    random_ut_loc = add_random_offset(default_ut_loc, max_offset_loc_ut)
    random_ut_angle = add_random_offset(default_ut_angle, max_offset_angle_ut)

    h_time = rt_channel_time(
        ut_loc=random_ut_loc,
        ut_angle=random_ut_angle,
    )
    return h_time


if __name__ == "__main__":
    from src.utils.plots import plot_channel_frequency_domain
    from sionna.channel import gen_single_sector_topology
    from sionna.channel.tr38901 import AntennaArray, UMi
    from src.settings.antenna import ut_array, bs_array, carrier_frequency
    import matplotlib.pyplot as plt
    
    h_freq = rt_channel_freq()
    plot_channel_frequency_domain(
        h_freq[0, 0, 0, 0, 0].numpy().real,
    )

    topology = gen_single_sector_topology(1, 1, "umi", min_ut_velocity=speed)
    ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology

    umi = UMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        enable_shadow_fading=False,
        enable_pathloss=False,
    )
    umi.set_topology(
        tf.constant([[default_ut_loc]], dtype=tf.float32),
        tf.constant([[default_bs_loc]], dtype=tf.float32),
        tf.constant([[default_ut_angle]], dtype=tf.float32),
        tf.constant([[default_bs_angle]], dtype=tf.float32),
        tf.constant(
            [[[speed / math.sqrt(2), speed / math.sqrt(2), 0]]], dtype=tf.float32
        ),
        in_state=tf.constant([[False]], dtype=tf.bool),
        los=False,
    )
    umi.show_topology()
    plt.show()

    ut_array.show()
    plt.show()
    bs_array.show()
    plt.show()
