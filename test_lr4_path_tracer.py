import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = pathlib.Path('/home/runner/work/MOI/MOI/lr4_path_tracer (2).py')
spec = importlib.util.spec_from_file_location('lr4_path_tracer', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


class TestPathTracerMethods(unittest.TestCase):
    def test_diffuse_previous_event_returns_hemisphere_direction(self):
        rng = np.random.default_rng(42)
        n = np.array([0.0, 0.0, 1.0])
        v = np.array([0.0, 0.0, 1.0])
        d = module.sample_next_direction(n, v, 'diffuse', rng)
        self.assertGreaterEqual(float(np.dot(d, n)), -1e-8)
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=7)

    def test_cook_torrance_method_produces_finite_color(self):
        c = module.shade_point(
            normal=np.array([0.0, 0.0, 1.0]),
            view_dir=np.array([0.0, 0.0, 1.0]),
            light_dir=np.array([0.2, 0.0, 1.0]),
            albedo=np.array([0.8, 0.7, 0.6]),
            roughness=0.4,
            metallic=0.2,
            method='cook_torrance',
        )
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertTrue(np.all(c >= 0.0))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            module.shade_point(
                normal=np.array([0.0, 0.0, 1.0]),
                view_dir=np.array([0.0, 0.0, 1.0]),
                light_dir=np.array([0.0, 0.0, 1.0]),
                albedo=np.array([1.0, 1.0, 1.0]),
                roughness=0.5,
                metallic=0.0,
                method='unknown',
            )


if __name__ == '__main__':
    unittest.main()
