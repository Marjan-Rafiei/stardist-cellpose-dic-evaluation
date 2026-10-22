import unittest

import numpy as np

from code.dataset_quality_audit import (
    connected_instance_ids,
    contiguous_instance_ids,
    instance_class_mapping_valid,
    raw_class_mapping_valid,
    stable_membership_hash,
)


class DatasetQualityAuditTests(unittest.TestCase):
    def test_membership_hash_is_order_independent(self) -> None:
        first = stable_membership_hash([("b", "2"), ("a", "1")])
        second = stable_membership_hash([("a", "1"), ("b", "2")])
        self.assertEqual(first, second)

    def test_instance_ids_must_be_contiguous_and_connected(self) -> None:
        valid = np.array([[0, 1, 1], [0, 0, 2], [0, 2, 2]], dtype=np.uint16)
        gap = np.array([[0, 1, 0], [0, 0, 3], [0, 3, 3]], dtype=np.uint16)
        disconnected = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint16)
        self.assertTrue(contiguous_instance_ids(valid))
        self.assertTrue(connected_instance_ids(valid))
        self.assertFalse(contiguous_instance_ids(gap))
        self.assertFalse(connected_instance_ids(disconnected))

    def test_raw_to_class_mapping_is_exact(self) -> None:
        raw = np.array([[0, 128], [255, 0]], dtype=np.uint8)
        classes = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        self.assertTrue(raw_class_mapping_valid(raw, classes, {0: 0, 128: 1, 255: 2}))
        classes[1, 0] = 1
        self.assertFalse(raw_class_mapping_valid(raw, classes, {0: 0, 128: 1, 255: 2}))

    def test_each_instance_has_one_nonbackground_class(self) -> None:
        instances = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
        classes = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
        self.assertTrue(instance_class_mapping_valid(instances, classes, {1, 2}))
        classes[0, 2] = 2
        self.assertFalse(instance_class_mapping_valid(instances, classes, {1, 2}))


if __name__ == "__main__":
    unittest.main()
