"""Scheduling isolation guards; no hardware or user settings are accessed."""
import unittest
from bench_cuda_route import configure_vulkan_schedule, VULKAN_SCHEDULE_KEYS, VULKAN_SCHEDULES


class ScheduleIsolationTests(unittest.TestCase):
    def test_all_candidates_clear_presence_based_inherited_flags(self):
        contaminated = {key: '0' for key in VULKAN_SCHEDULE_KEYS} | {'PATH': 'keep'}
        for candidate, active in VULKAN_SCHEDULES.items():
            with self.subTest(candidate=candidate):
                actual = configure_vulkan_schedule(contaminated, candidate)
                self.assertEqual({k: v for k, v in actual.items() if k != 'PATH'},
                                 {active: '1'} if active else {})
                self.assertEqual(actual['PATH'], 'keep')
                self.assertTrue(all(contaminated[k] == '0' for k in VULKAN_SCHEDULE_KEYS))

    def test_unknown_candidate_rejected(self):
        with self.assertRaises(ValueError):
            configure_vulkan_schedule({}, 'typo')


if __name__ == '__main__':
    unittest.main()
