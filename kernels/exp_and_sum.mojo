from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceildiv, exp, inf
from random import random_float64
from time import perf_counter_ns
from testing import assert_almost_equal
from memory import UnsafePointer
from utils.index import Index

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = 8
alias ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD
alias static_layout = Layout.row_major(UNKNOWN_VALUE, 1)


fn reduction_kernel[
    op: fn (Float32, Float32) -> Float32,
](
    input: LayoutTensor[mut=False, dtype, static_layout],
    output: LayoutTensor[mut=True, dtype, static_layout],
    initial_value: Float32,
) -> None:
    var shared = tb[dtype]().row_major[BLOCK_SIZE, 1]().shared().alloc()
    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc = initial_value
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < input.runtime_layout.dim(0):
            var val = input[global_idx, 0][0]
            acc = op(acc, val)

    shared[tid, 0] = acc
    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            var val1 = shared[tid, 0][0]
            var val2 = shared[tid + stride, 0][0]
            shared[tid, 0] = op(val1, val2)
        barrier()
        stride //= 2

    if tid == 0:
        output[block_idx.x, 0] = shared[0, 0]


fn exp_and_partial_sum_kernel(
    input: LayoutTensor[mut=False, dtype, static_layout],
    output: LayoutTensor[mut=True, dtype, static_layout],
    partial_sums: LayoutTensor[mut=True, dtype, static_layout],
    global_max: Float32,
) -> None:
    var shared = tb[dtype]().row_major[BLOCK_SIZE, 1]().shared().alloc()
    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var sum_acc: Float32 = 0.0
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < input.runtime_layout.dim(0):
            var input_val = input[global_idx, 0][0]
            var exp_val = exp(input_val - global_max)
            output[global_idx, 0] = exp_val
            sum_acc += exp_val

    shared[tid, 0] = sum_acc
    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            var val1 = shared[tid, 0][0]
            var val2 = shared[tid + stride, 0][0]
            shared[tid, 0] = val1 + val2
        barrier()
        stride //= 2

    if tid == 0:
        partial_sums[block_idx.x, 0] = shared[0, 0]


def reduce[
    op: fn (Float32, Float32) -> Float32
](
    ctx: DeviceContext,
    input_ptr: UnsafePointer[Float32],
    N: Int,
    initial_value: Float32,
) -> Float32:
    if N <= 0:
        return initial_value
    if N == 1:
        var result_buffer = ctx.enqueue_create_buffer[dtype](1)
        ctx.enqueue_copy(result_buffer.unsafe_ptr(), input_ptr, 1)
        ctx.synchronize()
        with result_buffer.map_to_host() as h:
            return h[0]

    alias kernel = reduction_kernel[op]
    var result_buffer = ctx.enqueue_create_buffer[dtype](1)
    var current_N = N
    var d_current = input_ptr

    while current_N > 1:
        var num_blocks = ceildiv(current_N, ELEMENTS_PER_BLOCK)
        var d_next: UnsafePointer[Float32]

        if num_blocks == 1:
            d_next = result_buffer.unsafe_ptr()
        else:
            var temp_buf = ctx.enqueue_create_buffer[dtype](num_blocks)
            d_next = temp_buf.unsafe_ptr()

        var runtime_in = RuntimeLayout[static_layout].row_major(
            Index(current_N, 1)
        )
        var runtime_out = RuntimeLayout[static_layout].row_major(
            Index(num_blocks, 1)
        )
        var input_tensor = LayoutTensor[mut=False, dtype, static_layout](
            d_current, runtime_in
        )
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            d_next, runtime_out
        )

        ctx.enqueue_function[kernel](
            input_tensor,
            output_tensor,
            initial_value,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
        d_current = d_next
        current_N = num_blocks

    ctx.synchronize()
    var final_result: Float32
    with result_buffer.map_to_host() as h:
        final_result = h[0]
    return final_result


def main():
    alias N = 1_000_000

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](N)
        var h_input_data = List[Float32](capacity=N)

        with input_buffer.map_to_host() as h_input:
            for i in range(N):
                var val = Float32(random_float64(-1.0, 1.0))
                h_input[i] = val
                h_input_data.append(val)

        fn max_op(a: Float32, b: Float32) -> Float32:
            return max(a, b)

        var global_max = reduce[max_op](
            ctx, input_buffer.unsafe_ptr(), N, -inf[dtype]()
        )

        var output_buffer = ctx.enqueue_create_buffer[dtype](N)
        var num_blocks = ceildiv(N, ELEMENTS_PER_BLOCK)
        var partial_sums_buffer = ctx.enqueue_create_buffer[dtype](num_blocks)

        var rt_layout_N = RuntimeLayout[static_layout].row_major(Index(N, 1))
        var rt_layout_partial = RuntimeLayout[static_layout].row_major(
            Index(num_blocks, 1)
        )
        var input_tensor = LayoutTensor[mut=False, dtype, static_layout](
            input_buffer.unsafe_ptr(), rt_layout_N
        )
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            output_buffer.unsafe_ptr(), rt_layout_N
        )
        var partial_sums_tensor = LayoutTensor[mut=True, dtype, static_layout](
            partial_sums_buffer.unsafe_ptr(), rt_layout_partial
        )

        ctx.enqueue_function[exp_and_partial_sum_kernel](
            input_tensor,
            output_tensor,
            partial_sums_tensor,
            global_max,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()

        var expected_numerators = List[Float32](capacity=N)
        for i in range(N):
            expected_numerators.append(exp(h_input_data[i] - global_max))

        with output_buffer.map_to_host() as h_output:
            for i in range(N):
                assert_almost_equal(
                    h_output[i], expected_numerators[i], rtol=1e-5
                )

        var expected_partial_sums = List[Float32](capacity=num_blocks)
        for i in range(num_blocks):
            var block_start = i * ELEMENTS_PER_BLOCK
            var block_end = min(block_start + ELEMENTS_PER_BLOCK, N)
            var current_sum: Float32 = 0.0
            for j in range(block_start, block_end):
                current_sum += expected_numerators[j]
            expected_partial_sums.append(current_sum)

        with partial_sums_buffer.map_to_host() as h_partial_sums:
            for i in range(num_blocks):
                assert_almost_equal(
                    h_partial_sums[i], expected_partial_sums[i], rtol=1e-3
                )
