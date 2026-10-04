import unittest

from disk_schedulers import schedule


class SchedulerTests(unittest.TestCase):
    def test_classic_upward_batch(self):
        requests = [82, 170, 43, 140, 24, 16, 190]
        expected = {
            "FCFS": ((100, 82, 170, 43, 140, 24, 16, 190), 628),
            "SCAN": ((100, 140, 170, 190, 199, 82, 43, 24, 16), 282),
            "C-SCAN": ((100, 140, 170, 190, 199, 0, 16, 24, 43, 82), 380),
            "SSTF": ((100, 82, 43, 24, 16, 140, 170, 190), 258),
        }
        for name, (path, seek) in expected.items():
            with self.subTest(name=name):
                result = schedule(name, requests)
                self.assertEqual(result.path, path)
                self.assertEqual(result.seek, seek)

    def test_downward_scan_and_wrap(self):
        self.assertEqual(schedule("SCAN", [20, 80, 150], 100, -1).path,
                         (100, 80, 20, 0, 150))
        self.assertEqual(schedule("C-SCAN", [20, 80, 150], 100, -1).path,
                         (100, 80, 20, 0, 199, 150))

    def test_no_unnecessary_endpoint_and_ties(self):
        self.assertEqual(schedule("SCAN", [110, 120]).seek, 20)
        self.assertEqual(schedule("C-SCAN", [110, 120]).seek, 20)
        self.assertEqual(schedule("SSTF", [90, 110], 100).path, (100, 90, 110))
        self.assertEqual(schedule("FCFS", []).seek, 0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            schedule("SCAN", [200])
        with self.assertRaises(ValueError):
            schedule("SCAN", [1], direction=0)


if __name__ == "__main__":
    unittest.main()
