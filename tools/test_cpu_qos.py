"""Pure mocked Win32 calls; no actual process policy is queried or changed."""
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_qos as qos


class FakeApi:
    def __init__(self, control=0, state=0, get_failure=None, set_failure=False, mismatch=False):
        self.control, self.state = control, state
        self.get_failure, self.set_failure, self.mismatch = get_failure, set_failure, mismatch
        self.gets, self.sets = [], []

    def GetProcessInformation(self, handle, info_class, pointer, size):
        self.gets.append((handle, info_class, size))
        if self.get_failure == len(self.gets):
            ctypes.set_last_error(50)
            return 0
        value = ctypes.cast(pointer, ctypes.POINTER(qos.PowerThrottlingState)).contents
        value.Version = 1
        value.ControlMask = self.control
        value.StateMask = self.state
        return 1

    def SetProcessInformation(self, handle, info_class, pointer, size):
        value = ctypes.cast(pointer, ctypes.POINTER(qos.PowerThrottlingState)).contents
        self.sets.append((handle, info_class, size, value.Version, value.ControlMask, value.StateMask))
        if self.set_failure:
            ctypes.set_last_error(5)
            return 0
        if not self.mismatch:
            self.control, self.state = value.ControlMask, value.StateMask
        return 1


class QosTests(unittest.TestCase):
    def invoke(self, api):
        with patch.object(qos, '_api', return_value=api):
            return qos.apply_high_qos(SimpleNamespace(_handle=9876, pid=1234))

    def test_system_managed_is_not_claimed_eco_and_exact_handle_used(self):
        api = FakeApi()
        value = self.invoke(api)
        self.assertTrue(value['applied_successfully'])
        self.assertIn('system_managed', value['before_interpretation'])
        self.assertFalse(value['no_op_already_high'])
        self.assertEqual(api.sets, [(9876, 4, 12, 1, 1, 0)])
        self.assertTrue(all(row[0] == 9876 for row in api.gets))
        self.assertGreaterEqual(value['setter_seconds'], 0)

    def test_explicit_high_is_readback_verified_no_op(self):
        api = FakeApi(control=1, state=0)
        value = self.invoke(api)
        self.assertTrue(value['applied_successfully']); self.assertTrue(value['no_op_already_high'])
        self.assertEqual(len(api.gets), 2); self.assertFalse(api.sets)
        self.assertIsNone(value['setter_seconds']); self.assertIsNone(value['setter_passed'])

    def test_timer_and_other_bits_preserved(self):
        api = FakeApi(control=0b1011, state=0b1011)
        value = self.invoke(api)
        self.assertTrue(value['applied_successfully'])
        self.assertEqual(api.sets[0][4:], (0b1011, 0b1010))
        self.assertEqual(value['after']['state_mask'] & ~1, 0b1010)

    def test_first_get_failure_never_sets(self):
        api = FakeApi(get_failure=1)
        value = self.invoke(api)
        self.assertFalse(value['applied_successfully']); self.assertFalse(api.sets)
        self.assertEqual(value['errors'][0]['win32_error'], 50)
        self.assertIsNone(value['before'])

    def test_set_failure_is_retained_even_with_readback(self):
        api = FakeApi(set_failure=True)
        value = self.invoke(api)
        self.assertFalse(value['applied_successfully']); self.assertFalse(value['setter_passed'])
        self.assertEqual(len(api.gets), 2)
        self.assertEqual(value['errors'][0]['win32_error'], 5)

    def test_after_get_failure_cannot_pass(self):
        value = self.invoke(FakeApi(get_failure=2))
        self.assertFalse(value['applied_successfully']); self.assertTrue(value['setter_passed'])
        self.assertIsNone(value['after'])

    def test_failed_readback_cannot_pass(self):
        value = self.invoke(FakeApi(mismatch=True))
        self.assertFalse(value['applied_successfully'])
        self.assertTrue(any(row['operation'] == 'Verify preserved masks' for row in value['errors']))

    def test_pid_or_invalid_handle_rejected_without_api_calls(self):
        with patch.object(qos, '_api') as api:
            for process in (1234, '1234', SimpleNamespace(pid=1234), SimpleNamespace(_handle=0)):
                self.assertFalse(qos.apply_high_qos(process)['applied_successfully'])
            api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
