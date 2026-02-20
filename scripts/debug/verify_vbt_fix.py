import sys
import os
import numpy as np
import vectorbt as vbt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from ggTrader.utils.vbt_patches import apply_vbt_patches


def test_fix():
    print("Applying patches...")
    apply_vbt_patches()

    print("Testing to_1d_array with list...")
    res = vbt.base.reshape_fns.to_1d_array([1, 2, 3])
    print(f"Result: {res}, Type: {type(res)}")

    print("Testing to_1d_array with 2D array...")
    arr2d = np.array([[1], [2], [3]])
    res2 = vbt.base.reshape_fns.to_1d_array(arr2d)
    print(f"Result 2D: {res2}, Shape: {res2.shape}")

    print("Success!")


if __name__ == "__main__":
    test_fix()
