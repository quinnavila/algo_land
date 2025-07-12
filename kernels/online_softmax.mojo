from time import perf_counter_ns
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from math import ceildiv, exp, inf
from random import random_float64
from layout import Layout, LayoutTensor

alias dtype = DType.float32

alias BATCH = 1
alias M_DIM = 16
alias N_DIM = 1024

alias BLOCK_M = M_DIM
alias BLOCK_N = 256
alias N_STEPS = N_DIM // BLOCK_N

alias TPB = 256

alias scores_tile_layout = Layout.row_major(BLOCK_M, BLOCK_N)
alias output_layout = Layout.row_major(M_DIM, N_DIM)
alias row_stats_layout = Layout.row_major(M_DIM, 2)


fn online_softmax_update_kernel[
    scores_tile_layout: Layout,
    output_layout: Layout,
    row_stats_layout: Layout,
](
    output: LayoutTensor[mut=True, dtype, output_layout],
    scores_tile: LayoutTensor[mut=False, dtype, scores_tile_layout],
    row_stats: LayoutTensor[mut=True, dtype, row_stats_layout],
    tile_col_offset: Int,
):
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= M_DIM:
        return

    var old_max = row_stats[row, 0][0]
    var old_sum = row_stats[row, 1][0]

    var tile_max: Float32 = -inf[dtype]()
    for j in range(BLOCK_N):
        var val = scores_tile[row, j][0]
        if val > tile_max:
            tile_max = val

    var new_max = old_max
    if tile_max > old_max:
        new_max = tile_max

    var scale = exp(old_max - new_max)
    var new_sum = old_sum * scale

    for j in range(tile_col_offset):
        output[row, j] *= scale

    for j in range(BLOCK_N):
        var val = scores_tile[row, j][0]
        var exp_val = exp(val - new_max)
        new_sum += exp_val
        output[row, tile_col_offset + j] = exp_val

    row_stats[row, 0] = new_max
    row_stats[row, 1] = new_sum


fn normalize_kernel[
    output_layout: Layout,
    row_stats_layout: Layout,
](
    output: LayoutTensor[mut=True, dtype, output_layout],
    row_stats: LayoutTensor[mut=False, dtype, row_stats_layout],
):
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= M_DIM:
        return

    var total_sum = row_stats[row, 1][0]
    if total_sum > 1e-9:
        var inv_sum = 1.0 / total_sum
        for j in range(N_DIM):
            output[row, j] *= inv_sum


def main():
    print("Running Fused Online Softmax (Simplified)")
    print("Matrix size (M x N):", M_DIM, "x", N_DIM)
    print("Processing in", N_STEPS, "steps of tile size", BLOCK_N)

    with DeviceContext() as ctx:
        var scores_tile_buffer = ctx.enqueue_create_buffer[dtype](
            scores_tile_layout.size()
        )
        var output_buffer = ctx.enqueue_create_buffer[dtype](
            output_layout.size()
        )
        var row_stats_buffer = ctx.enqueue_create_buffer[dtype](
            row_stats_layout.size()
        )

        with row_stats_buffer.map_to_host() as h_stats:
            var stats_tensor = LayoutTensor[dtype, row_stats_layout](
                h_stats.unsafe_ptr()
            )
            for i in range(M_DIM):
                stats_tensor[i, 0] = -inf[dtype]()
                stats_tensor[i, 1] = 0.0

        var output_tensor = LayoutTensor[dtype, output_layout](output_buffer)
        var row_stats_tensor = LayoutTensor[dtype, row_stats_layout](
            row_stats_buffer
        )

        var blocks_per_grid = ceildiv(M_DIM, TPB)
        var start_time_ns = perf_counter_ns()

        for i in range(N_STEPS):
            var tile_col_offset = i * BLOCK_N

            with scores_tile_buffer.map_to_host() as h_scores:
                var scores_tensor = LayoutTensor[dtype, scores_tile_layout](
                    h_scores.unsafe_ptr()
                )
                for r in range(BLOCK_M):
                    for c in range(BLOCK_N):
                        scores_tensor[r, c] = Float32(random_float64(-2.0, 2.0))

            var scores_tile_tensor = LayoutTensor[dtype, scores_tile_layout](
                scores_tile_buffer
            )

            ctx.enqueue_function[
                online_softmax_update_kernel[
                    scores_tile_layout, output_layout, row_stats_layout
                ]
            ](
                output_tensor,
                scores_tile_tensor,
                row_stats_tensor,
                tile_col_offset,
                grid_dim=blocks_per_grid,
                block_dim=TPB,
            )

        ctx.enqueue_function[normalize_kernel[output_layout, row_stats_layout]](
            output_tensor,
            row_stats_tensor,
            grid_dim=blocks_per_grid,
            block_dim=TPB,
        )

        ctx.synchronize()
        var end_time_ns = perf_counter_ns()
        var duration_ms = (end_time_ns - start_time_ns) / 1_000_000

        print("\nOnline softmax computation complete.")
        print("Execution time:", duration_ms, "ms")

        with output_buffer.map_to_host() as h_output:
            var result_tensor = LayoutTensor[dtype, output_layout](
                h_output.unsafe_ptr()
            )
            var row_sum: Float32 = 0.0
            for j in range(N_DIM):
                row_sum += result_tensor[0, j][0]

            print("Sum of the first row:", row_sum)
            if row_sum > 0.999 and row_sum < 1.001:
                print("Verification PASSED.")
            else:
                print("Verification FAILED.")
