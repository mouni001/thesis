import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from stream_annotations import INSECTS_ANNOTATIONS, get_stream_annotation  # noqa: E402


class StreamAnnotationTests(unittest.TestCase):
    def test_all_points_are_inside_their_stream(self):
        for name, annotation in INSECTS_ANNOTATIONS.items():
            with self.subTest(dataset=name):
                n = annotation["instances"]
                for point in annotation["exact_abrupt_points"]:
                    self.assertGreater(point, 0)
                    self.assertLess(point, n)
                for point in annotation["reference_points"]:
                    self.assertGreater(point, 0)
                    self.assertLess(point, n)

    def test_incremental_stream_is_not_given_fake_abrupt_truth(self):
        annotation = get_stream_annotation("INSECTS_incremental_imbalanced.csv")
        self.assertEqual(annotation["change_pattern"], "incremental_throughout")
        self.assertEqual(annotation["exact_abrupt_points"], [])

    def test_abrupt_stream_has_exact_boundaries_and_source(self):
        annotation = get_stream_annotation("/tmp/INSECTS_abrupt_balanced.csv")
        self.assertEqual(
            annotation["exact_abrupt_points"],
            [14352, 19500, 33240, 38682, 39510],
        )
        self.assertIn("doi", annotation["source"])


if __name__ == "__main__":
    unittest.main()
