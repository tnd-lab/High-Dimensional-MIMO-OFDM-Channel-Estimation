from sionna.utils import BinarySource

from src.settings.config import batch_size, num_streams_per_tx, num_tx, number_of_bits


def binary_sources(matrix_size):
    binary_source = BinarySource()
    binary_values = binary_source(list(matrix_size))
    return binary_values


if __name__ == "__main__":
    binary_values = binary_sources(
        [batch_size, num_tx, num_streams_per_tx, number_of_bits]
    )

    print(f"Binary values: {binary_values}")
