from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from machivellian.convert import (convert_d_to_r,
                                  convert_r_to_d,
                                  convert_or_to_d,
                                  convert_d_to_or,
                                  )


class PowerConvertTest(TestCase):
    def setUp(self):
        self.r = 0.5000
        self.sr = 0.0058

        self.d1 = 1.1547
        self.sd1 = 0.0550

        self.d2 = 0.5000
        self.sd2 = 0.0205

        self.lor = 0.9069
        self.slor = 0.0676

    def test_convert_d_to_r(self):
        r, sr = convert_d_to_r(self.d1)
        npt.assert_almost_equal(self.r, r, 4)
        self.assertTrue(np.isnan(sr))

    def test_convert_d_to_r_error(self):
        _, sr = convert_d_to_r(self.d1, self.sd1)
        npt.assert_almost_equal(sr, self.sr, 4)

    def test_convert_r_to_d(self):
        d, sd = convert_r_to_d(self.r)
        npt.assert_almost_equal(d, self.d1, 4)
        self.assertTrue(np.isnan(sd))

    def test_convert_r_to_d_error(self):
        _, sd = convert_r_to_d(self.r, self.sr)
        npt.assert_almost_equal(sd, self.sd1, 4)

    def test_convert_or_to_d(self):
        d, sd = convert_or_to_d(self.lor)
        npt.assert_almost_equal(d, self.d2, 4)
        self.assertTrue(np.isnan(sd))

    def test_convert_or_to_d_error(self):
        _, sd = convert_or_to_d(self.lor, self.slor)
        npt.assert_almost_equal(sd, self.sd2, 4)

    def test_convert_d_to_or(self):
        lor, slor = convert_d_to_or(self.d2)
        npt.assert_almost_equal(lor, self.lor, 4)
        self.assertTrue(np.isnan(slor))

    def test_convert_d_to_or_error(self):
        _, slor = convert_d_to_or(self.d2, self.sd2)
        npt.assert_almost_equal(slor, self.slor, 3)


if __name__ == '__main__':
    main()
