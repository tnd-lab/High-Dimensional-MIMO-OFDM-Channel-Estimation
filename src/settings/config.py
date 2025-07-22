import numpy as np

subcarrier_spacing = 15e3
max_resource_block = 79
num_resource_block = min(5, max_resource_block)
num_subcarrier = num_resource_block * 12
num_ofdm_symbols = 14
fft_size = num_subcarrier

batch_size = 1
num_guards = [5, 6]
dc_null = True
num_tx = 1
num_rx = 1
rx_tx_association = np.array([[1]])  # because 1 tx and 1 rx
pilots = [2, 11]
cyclic_prefix_length = 6
pilot_pattern = "kronecker"
bandwidth = num_subcarrier * subcarrier_spacing

code_rate = 0.5
bits_per_symbol = 2
num_effective_subcarriers = num_subcarrier - sum(num_guards) - 1
number_of_bits = (num_subcarrier - sum(num_guards) - 1) * (
    num_ofdm_symbols - len(pilots)
)

######################
#  Antenna settings  #
######################
num_ut_ant = 2
num_bs_ant = 4
num_streams_per_tx = min(num_bs_ant, num_ut_ant)
carrier_frequency = 2.6e9

##########################
#  CDL Channel settings  #
##########################

# Nominal delay spread in [s]. Please see the CDL documentation
# about how to choose this value.
delay_spread = 100e-9

# The `direction` determines if the UT or BS is transmitting.
# In the `uplink`, the UT is transmitting. "Downlink", The BS is transmitting
direction = "uplink"

# Suitable values are ["A", "B", "C", "D", "E"]
cdl_model = "B"

# UT speed [m/s]. BSs are always assumed to be fixed.
# The direction of travel will chosen randomly within the x-y plane.
speed = 0

ebno_db = 0
