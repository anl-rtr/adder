import pytest
import os
import numpy as np
from tests.testing_harness import TestHarness

inp_file = """Pin cell problem
1 1 -10.2570 -71 1 -2 imp:n=1
2 2 -10.2570  71 -72  1 -2         imp:n=1 vol=2.
3 3 -10.2570  72 -73  1 -2         imp:n=1
4 4 -1.79E-4  73  -8  1 -2         imp:n=1
5 5 -6.5600    8  -9  1 -2         imp:n=1
6 6 -0.700     9   1 -2  3 -4 5 -6 imp:n=1
7 0 #(1 -2 3 -4 5 -6) imp:n=0

*1 pz -0.9486374
*2 pz 0.9486374
*3 px -0.63
*4 px 0.63
*5 py -0.63
*6 py 0.63
71 cz 0.236482670260071
72 cz 0.334436999547997
73 cz 0.4096
8 cz 0.4180
9 cz 0.4750

m0 nlib=70c
m1  92235.70c  0.1 92238.70c  0.1 8016.70c  0.8
m2  92235.70c -0.285555944474272
     92238.70c -0.578417982008341
     8016.70c -0.136026073517386 nlib=70c
m3  92235.70c  0.2 92238.70c  0.1 8016.70c  0.7 nlib=70c
m4  92235.70c  0.2 92238.70c  0.2 8016.70c  0.6 nlib=70c
m5  92235.70c  0.3 92238.70c  0.1 8016.70c  0.6 nlib=70c
m6  92235.70c  0.3 92238.70c  0.2 8016.70c  0.5 nlib=70c
ksrc -0.20 0 0 0.20 0.20 0 0 -0.20 0 0 0 0 0
     -0.5 0 0 0.5
kcode 1000 1.18 20 300
print 38 60 128 130
void  
cut:n 2E7
lost 10 1
nps 15

"""


class VolumeHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)

    def _cleanup(self):
        super()._cleanup()
        try:
            os.remove("v_calco")
        except OSError:
            pass

    def _write_results(self, results_string):
        pass

    def _overwrite_results(self):
        pass

    def _get_results(self):
        pass

    def _compare_results(self):
        ref_vols = np.array([0.333333, 2., 0.333333, 1., 0.303394,
                             1.667284])
        ref_tols = np.array([0., 0., 0., 0., 0., 0.2 / 100.])

        # Get the outputs
        new_inp = self._get_outputs().split("\n")

        # Get the volumes
        num_vols = 6
        test_vols = []
        for line in new_inp[3: 3 + num_vols]:
            test_vols.append(float(line.split("vol=")[1].split()[0]))

        # Now we can test them
        for i in range(len(ref_vols)):
            if ref_tols[i] == 0.:
                assert ref_vols[i] == test_vols[i]
            else:
                assert (abs(ref_vols[i] - test_vols[i]) / ref_vols[i]
                        < ref_tols[i])


def test_mcnp_volume():
    # This tests MCNPs ability to perform volume calcs

    output_text_files = ["state{}.inp".format(i + 1) for i in range(1)]
    test = VolumeHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()
