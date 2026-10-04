import unittest

import numpy as np

from disk_schedulers import SCHEDULERS, schedule
from experiment import dataset, make_split, make_timeline, seek_values


class ExperimentTests(unittest.TestCase):
    def test_workloads_are_reproducible_and_valid(self):
        first = make_split(12345, 30)
        self.assertEqual(first, make_split(12345, 30))
        self.assertNotEqual(first, make_split(12346, 30))
        for item in first:
            self.assertEqual(len(item["requests"]), 20)
            self.assertTrue(all(0 <= r <= 199 for r in item["requests"]))
        timeline = make_timeline(23456)
        self.assertEqual([item["kind"] for item in timeline[:30]], ["sequential"] * 30)
        self.assertEqual([item["kind"] for item in timeline[30:]], ["bursty"] * 30)

    def test_labels_really_minimize_seek(self):
        items = make_split(44, 12)
        _, seeks, labels = dataset(items)
        for item, seek_row, label in zip(items, seeks, labels):
            self.assertEqual(seek_row[label], min(seek_row))
            self.assertEqual(seek_values(item["requests"])[label],
                             schedule(SCHEDULERS[label], item["requests"]).seek)
        self.assertEqual(seeks.shape, (12, 4))
        self.assertTrue(np.issubdtype(seeks.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
