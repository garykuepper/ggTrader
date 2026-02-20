import unittest
import numpy as np
import vectorbt.base.reshape_fns as reshape_fns
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import the module under test
from ggTrader.utils.vbt_patches import apply_vbt_patches, _patched_to_1d_array


class TestVBTPatches(unittest.TestCase):
    def setUp(self):
        # Stash the real original function so we can restore it
        if not hasattr(reshape_fns, "_real_orig_to_1d_array"):
            # Check if current function is our patch
            current_func = reshape_fns.to_1d_array
            is_patched = False

            # Check by identity
            if current_func == _patched_to_1d_array:
                is_patched = True
            # Check by name if possible (regular functions)
            elif (
                hasattr(current_func, "__name__")
                and current_func.__name__ == "_patched_to_1d_array"
            ):
                is_patched = True

            if is_patched:
                # It's already patched. try to find original.
                if hasattr(reshape_fns, "_orig_to_1d_array"):
                    self.real_func = reshape_fns._orig_to_1d_array
                else:
                    # fallback
                    self.real_func = lambda x: np.asarray(x).flatten()
            else:
                self.real_func = reshape_fns.to_1d_array
        else:
            self.real_func = reshape_fns._real_orig_to_1d_array

    def tearDown(self):
        # Restore the module-level function to a safe state
        reshape_fns.to_1d_array = self.real_func
        # Remove our stash if we put it there
        if hasattr(reshape_fns, "_orig_to_1d_array"):
            del reshape_fns._orig_to_1d_array

    def test_fix_name_error_condition(self):
        """
        Test that _patched_to_1d_array handles the case where the 'original' function
        raises a NameError (simulating a stale patch referencing a deleted global).
        """

        # 1. Simulate the "Broken" Function
        # This represents the old patch that tries to call 'orig_to_1d_array'
        # which no longer exists.
        def broken_old_patch(*args, **kwargs):
            raise NameError("name 'orig_to_1d_array' is not defined")

        # 2. Simulate the "Dirty" State
        # The module has this broken function stored as the "original"
        # because the user reloaded the module but the patch logic preserved
        # the (now broken) function reference.
        reshape_fns._orig_to_1d_array = broken_old_patch

        # 3. Apply our NEW patch
        # This sets reshape_fns.to_1d_array = _patched_to_1d_array
        # and it should see _orig_to_1d_array is already there (our broken one)
        apply_vbt_patches()

        # Verify our patch is active
        self.assertEqual(reshape_fns.to_1d_array.__name__, "_patched_to_1d_array")

        # 4. Trigger the logic
        # Calling to_1d_array should:
        #   -> Call _patched_to_1d_array
        #   -> Try calling reshape_fns._orig_to_1d_array (which is broken_old_patch)
        #   -> Catch NameError
        #   -> Fallback to _clean_to_1d_array
        #   -> Return correct result

        input_arr = np.array([[1], [2], [3]])
        # This would crash if the fix wasn't working
        result = reshape_fns.to_1d_array(input_arr)

        # 5. Assertions
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.ndim, 1)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

        print("\nTest passed: Successfully recovered from NameError in stale patch.")


if __name__ == "__main__":
    unittest.main()
