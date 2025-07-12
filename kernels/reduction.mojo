from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceildiv, inf
from random import random_float64
from time import perf_counter_ns
from testing import assert_almost_equal
from memory import UnsafePointer
from utils.index import Index

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = 8
alias ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD


fn reduction_kernel[
    op: fn (Float32, Float32) -> Float32,
](
    input: LayoutTensor[mut=False, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    output: LayoutTensor[mut=True, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
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
        output[0, block_idx.x] = shared[0, 0]


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

    alias static_layout = Layout.row_major(UNKNOWN_VALUE, 1)
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

        var runtime_in_layout = RuntimeLayout[static_layout].row_major(
            Index(current_N, 1)
        )
        var runtime_out_layout = RuntimeLayout[static_layout].row_major(
            Index(num_blocks, 1)
        )

        var input_tensor = LayoutTensor[mut=False, dtype, static_layout](
            d_current, runtime_in_layout
        )
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            d_next, runtime_out_layout
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
    alias N_initial = 1_000_000

    print("Step 1: Building a Reusable, Scalable Reduction Function")
    print("Array size (N):", N_initial)
    print("-" * 60)

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](N_initial)
        var h_input_data = List[Float32]()
        h_input_data.reserve(N_initial)

        with input_buffer.map_to_host() as h_input:
            for i in range(N_initial):
                var val = Float32(random_float64(-10.0, 10.0))
                h_input[i] = val
                h_input_data.append(val)

        fn add_op(a: Float32, b: Float32) -> Float32:
            return a + b

        fn max_op(a: Float32, b: Float32) -> Float32:
            return max(a, b)

        print("Test Case 1: Sum Reduction")
        var start_time_sum = perf_counter_ns()
        var gpu_sum = reduce[add_op](
            ctx, input_buffer.unsafe_ptr(), N_initial, 0.0
        )
        var end_time_sum = perf_counter_ns()

        var cpu_sum: Float32 = 0.0
        for i in range(N_initial):
            cpu_sum += h_input_data[i]

        print("GPU Sum:", gpu_sum)
        print("CPU Sum:", cpu_sum)
        print("Time:", (end_time_sum - start_time_sum) / 1_000_000, "ms")
        assert_almost_equal(gpu_sum, cpu_sum, rtol=1e-3)
        print("Verification PASSED.\n")

        print("Test Case 2: Max Reduction")
        var start_time_max = perf_counter_ns()
        var gpu_max = reduce[max_op](
            ctx, input_buffer.unsafe_ptr(), N_initial, -inf[dtype]()
        )
        var end_time_max = perf_counter_ns()

        var cpu_max: Float32 = -inf[dtype]()
        for i in range(N_initial):
            if h_input_data[i] > cpu_max:
                cpu_max = h_input_data[i]

        print("GPU Max:", gpu_max)
        print("CPU Max:", cpu_max)
        print("Time:", (end_time_max - start_time_max) / 1_000_000, "ms")
        assert_almost_equal(gpu_max, cpu_max)
        print("Verification PASSED.")
