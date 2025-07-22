from sionna.ofdm import ResourceGrid, ResourceGridMapper

from src.data.binary_sources import binary_sources
from src.data.qam_mapper import qam_mapper
from src.ldpc.ldpc_encoder import ldpc_encoder
from src.settings.config import (
    batch_size,
    bits_per_symbol,
    code_rate,
    cyclic_prefix_length,
    dc_null,
    num_guards,
    num_ofdm_symbols,
    num_streams_per_tx,
    num_subcarrier,
    num_tx,
    number_of_bits,
    pilot_pattern,
    pilots,
    subcarrier_spacing,
)
from src.utils.plots import plot_symbols

rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=num_subcarrier,
    subcarrier_spacing=subcarrier_spacing,
    num_tx=num_tx,
    num_streams_per_tx=num_streams_per_tx,
    cyclic_prefix_length=cyclic_prefix_length,
    num_guard_carriers=num_guards,
    dc_null=dc_null,
    pilot_pattern=pilot_pattern,
    pilot_ofdm_symbol_indices=pilots,
)


def resource_grid_mapper(symbols):
    rg_mapper = ResourceGridMapper(rg)

    mapped_symbols = rg_mapper(symbols)
    return mapped_symbols


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rg.show()
    plt.savefig("./images/resource_grid.png")

    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )
    encoded_binary_values = ldpc_encoder(binary_values)
    qam_symbols = qam_mapper(encoded_binary_values)
    mapped_qam_symbol = resource_grid_mapper(qam_symbols)

    print(f"Binary values: {binary_values}")
    print(f"Encoded binary values: {encoded_binary_values}")
    print(f"QAM symbols: {qam_symbols}")
    print(f"Mapped QAM symbols: {mapped_qam_symbol}")
    plot_symbols(
        mapped_qam_symbol[0, 0, 0].numpy().real.T,
        save_fig="./images/mapped_qam_symbol.png",
    )
