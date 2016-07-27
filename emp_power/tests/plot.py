from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from emp_power.plot import (_set_ticks,
                            _get_symetrical,
                            )


class PlotTest(TestCase):
    def setup(self):
        pass

    def test_set_ticks(self):
        known = np.arange(0, 1.1, 0.25)
        test = _set_ticks([0, 1], 0.25)
        print(test)
        # npt.assert_array_equal(known, test)

    def test_get_symetrical(self):
        known = [-5, 5]
        self.assertEqual(known, _get_symetrical([-2, 5]))
        self.assertEqual(known, _get_symetrical(known))

if __name__ == '__main__':
    main()
