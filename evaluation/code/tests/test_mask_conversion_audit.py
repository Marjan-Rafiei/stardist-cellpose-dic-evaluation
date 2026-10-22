import unittest

import numpy as np

from code.mask_conversion_audit import (
    convert_mask,
    faulty_unconditional_uint8,
    prepare_mask_uint8,
)


class MaskConversionAuditTests(unittest.TestCase):
    def test_faulty_uint8_mapping_reproduces_overflow(self):
        values = np.asarray([0, 128, 255], dtype=np.uint8)
        self.assertEqual(faulty_unconditional_uint8(values).tolist(), [0, 128, 1])

    def test_corrected_branch_preserves_uint8_masks(self):
        values = np.asarray([0, 128, 255], dtype=np.uint8)
        self.assertEqual(prepare_mask_uint8(values).tolist(), [0, 128, 255])

    def test_corrected_branch_rescales_normalized_masks(self):
        values = np.asarray([0.0, 128.0 / 255.0, 1.0], dtype=np.float64)
        self.assertEqual(prepare_mask_uint8(values).tolist(), [0, 128, 255])

    def test_corrected_conversion_separates_classes_and_instances(self):
        mask = np.asarray(
            [
                [128, 0, 255],
                [128, 0, 255],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        instances, classes, n_interphase, n_mitotic = convert_mask(mask)
        self.assertEqual((n_interphase, n_mitotic), (1, 1))
        self.assertEqual(np.unique(instances).tolist(), [0, 1, 2])
        self.assertEqual(classes.tolist(), [[1, 0, 2], [1, 0, 2], [0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
