# File: batched_col.mojo

from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from math import ceildiv, exp, inf
from random import random_float64
from time import perf_counter_ns
from testing import assert_almost_equal

alias dtype = DType.float32
alias ROWS = 128
alias COLS = 1024
alias BLOCK_SIZE_X = 32
alias BLOCK_SIZE_Y = 8

alias static_layout = Layout.row_major(ROWS, COLS)
alias col_stats_layout = Layout.row_major(1, COLS)


fn col_max_kernel(
    input: LayoutTensor[mut=False, dtype, static_layout],
    col_maxes: LayoutTensor[mut=True, dtype, col_stats_layout],
) -> None:
    var col = block_idx.x * block_dim.x + thread_idx.x
    if col >= COLS:
        return

    var max_val = -inf[dtype]()
    for r in range(ROWS):
        max_val = max(max_val, input[r, col][0])

    col_maxes[0, col] = max_val


fn col_sum_kernel(
    input: LayoutTensor[mut=False, dtype, static_layout],
    col_maxes: LayoutTensor[mut=False, dtype, col_stats_layout],
    col_sums: LayoutTensor[mut=True, dtype, col_stats_layout],
) -> None:
    var col = block_idx.x * block_dim.x + thread_idx.x
    if col >= COLS:
        return

    var max_val = col_maxes[0, col][0]
    var sum_val: Float32 = 0.0
    for r in range(ROWS):
        sum_val += exp(input[r, col][0] - max_val)

    col_sums[0, col] = sum_val


fn col_normalize_kernel(
    input: LayoutTensor[mut=False, dtype, static_layout],
    output: LayoutTensor[mut=True, dtype, static_layout],
    col_maxes: LayoutTensor[mut=False, dtype, col_stats_layout],
    col_sums: LayoutTensor[mut=False, dtype, col_stats_layout],
) -> None:
    var col = block_idx.x * block_dim.x + thread_idx.x
    var row = block_idx.y * block_dim.y + thread_idx.y

    if row < ROWS and col < COLS:
        var max_val = col_maxes[0, col][0]
        var sum_val = col_sums[0, col][0]
        if sum_val > 1e-9:
            var exp_val = exp(input[row, col][0] - max_val)
            output[row, col] = exp_val / sum_val


def main():
    print("Batched Column-wise Softmax (Multi-Pass)")
    print("Matrix size:", ROWS, "x", COLS)

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](ROWS * COLS)
        var output_buffer = ctx.enqueue_create_buffer[dtype](ROWS * COLS)

        with input_buffer.map_to_host() as h_input:
            for i in range(ROWS * COLS):
                h_input[i] = Float32(random_float64(-1.0, 1.0))

        var input_tensor = LayoutTensor[mut=False, dtype, static_layout](
            input_buffer.unsafe_ptr()
        )
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            output_buffer.unsafe_ptr()
        )

        var col_max_buffer = ctx.enqueue_create_buffer[dtype](COLS)
        var col_sum_buffer = ctx.enqueue_create_buffer[dtype](COLS)
        var col_max_tensor = LayoutTensor[mut=True, dtype, col_stats_layout](
            col_max_buffer.unsafe_ptr()
        )
        var col_sum_tensor = LayoutTensor[mut=True, dtype, col_stats_layout](
            col_sum_buffer.unsafe_ptr()
        )

        var grid_dim_1d = (ceildiv(COLS, BLOCK_SIZE_X), 1)
        var block_dim_1d = (BLOCK_SIZE_X, 1)

        print("\nLaunching 3-pass column-wise kernels...")
        var start_time = perf_counter_ns()
        ctx.enqueue_function[col_max_kernel](
            input_tensor,
            col_max_tensor,
            grid_dim=grid_dim_1d,
            block_dim=block_dim_1d,
        )
        ctx.enqueue_function[col_sum_kernel](
            input_tensor,
            col_max_tensor,
            col_sum_tensor,
            grid_dim=grid_dim_1d,
            block_dim=block_dim_1d,
        )

        var grid_dim_2d = (
            ceildiv(COLS, BLOCK_SIZE_X),
            ceildiv(ROWS, BLOCK_SIZE_Y),
        )
        var block_dim_2d = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
        ctx.enqueue_function[col_normalize_kernel](
            input_tensor,
            output_tensor,
            col_max_tensor,
            col_sum_tensor,
            grid_dim=grid_dim_2d,
            block_dim=block_dim_2d,
        )

        ctx.synchronize()
        var end_time = perf_counter_ns()
        print(
            "Total kernel execution time:",
            (end_time - start_time) / 1_000_000,
            "ms",
        )

        print("\nVerifying results...")
        with output_buffer.map_to_host() as h_output:
            var gpu_tensor = LayoutTensor[mut=False, dtype, static_layout](
                h_output.unsafe_ptr()
            )
            for c in range(COLS):
                var col_sum: Float32 = 0.0
                for r in range(ROWS):
                    col_sum += gpu_tensor[r, c][0]
                assert_almost_equal(col_sum, 1.0, rtol=1e-3)

        print("Verification PASSED: All columns sum to 1.0.")
