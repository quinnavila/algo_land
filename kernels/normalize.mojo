from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from math import ceildiv
from random import random_float64
from testing import assert_almost_equal
from memory import UnsafePointer
from utils.index import Index

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias static_layout = Layout.row_major(UNKNOWN_VALUE, 1)


fn normalize_kernel(
    output: LayoutTensor[mut=True, dtype, static_layout],
    global_sum: Float32,
) -> None:
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < output.runtime_layout.dim(0):
        output[idx, 0] = output[idx, 0][0] / global_sum


def main():
    alias N = 1_000_000
    print("Step 3: Testing the Final Normalization Kernel")
    print("Array size (N):", N)

    with DeviceContext() as ctx:
        var numerators_buffer = ctx.enqueue_create_buffer[dtype](N)
        var h_numerators = List[Float32](capacity=N)

        with numerators_buffer.map_to_host() as h_output:
            for i in range(N):
                var val = Float32(random_float64(0.1, 5.0))
                h_output[i] = val
                h_numerators.append(val)

        var global_sum = Float32(1234.56)

        var rt_layout = RuntimeLayout[static_layout].row_major(Index(N, 1))
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            numerators_buffer.unsafe_ptr(), rt_layout
        )

        print("\nLaunching normalize_kernel...")
        var num_blocks = ceildiv(N, BLOCK_SIZE)
        ctx.enqueue_function[normalize_kernel](
            output_tensor,
            global_sum,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()
        print("Kernel launch complete.")

        print("\nVerifying results...")
        var expected_results = List[Float32](capacity=N)
        for i in range(N):
            expected_results.append(h_numerators[i] / global_sum)

        with numerators_buffer.map_to_host() as h_output:
            for i in range(N):
                assert_almost_equal(h_output[i], expected_results[i], rtol=1e-6)

        print("Verification PASSED.")
