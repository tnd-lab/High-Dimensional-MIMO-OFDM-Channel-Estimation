import matplotlib.pyplot as plt
from sionna.channel import gen_single_sector_topology
from sionna.channel.tr38901 import AntennaArray, UMi

from src.settings.config import carrier_frequency, num_bs_ant, num_ut_ant, speed

ut_array = AntennaArray(
    num_rows=1,
    num_cols=1 if num_ut_ant == 1 else int(num_ut_ant // 2),
    polarization="single" if num_ut_ant == 1 else "dual",
    polarization_type="V" if num_ut_ant == 1 else "cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)


bs_array = AntennaArray(
    num_rows=1,
    num_cols=1 if num_bs_ant == 1 else int(num_bs_ant // 2),
    polarization="single" if num_bs_ant == 1 else "dual",
    polarization_type="V" if num_bs_ant == 1 else "cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)


if __name__ == "__main__":
    # UT array
    ut_array.show()
    ut_array.show_element_radiation_pattern()
    plt.show()

    # BS array
    bs_array.show()
    bs_array.show_element_radiation_pattern()
    plt.show()

    topology = gen_single_sector_topology(1, 1, "umi", min_ut_velocity=speed)
    ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology

    umi = UMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        enable_shadow_fading=False,
        enable_pathloss=True,
    )
    umi.set_topology(
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
    )
    umi.show_topology()
    plt.show()
