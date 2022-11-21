import numpy as np
from collections import OrderedDict

from adder import DepletionLibrary
from adder.origen22 import Origen22Depletion
from adder.constants import ORIGEN_ISO_DECAY_TYPES
from adder.origen22.depletionlibrary_origen import to_origen

origen_lib = \
"""   1    TEST DECAY LIBRARY: ACTIVATION
   1   10010  6,0.0, 0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       9.998E 01 1.000E 00 1.000E 00
   1   10020  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       1.557E-02 1.000E 00 1.000E 00
   1   20040  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       1.000E 02 1.000E 00 1.000E 00
   1   70160  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       0.0       3.000E-08 1.000E 00
  -1
   2    TEST DECAY LIBRARY: ACTINIDE
   2  902310  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       2.010E 02 0.0       4.000E-08 2.000E-04
   2  922350  1     1.386E 00 0.0       0.0       0.0       1.000E 00 0.0
   2                0.0       0.0       2.000E 02 7.204E-01 4.000E-12 3.000E-05
   2  922380  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       2.020E 02 9.927E 01 3.000E-12 4.000E-05
  -1
   3    TEST DECAY LIBRARY: FP
   3   10010  6     0.0       0.0       0.0       0.0       0.0       0.0
   3                0.0       0.0       0.0       9.998E 01 1.000E 00 1.000E 00
   3   80160  1     3.465E-01 0.0       1.000E 00 0.0       0.0       0.0
   3                0.0       0.0       0.0       9.976E 01 1.000E 00 1.000E 00
  -1
   4    TEST XS AND YIELD LIBRARY: ACTIVATION
  -1
   5    TEST XS AND YIELD LIBRARY: ACTINIDE
   5  922350 0.0       0.0       0.0       1.000E 02 0.0       0.0         -1.0
   5  922380 0.0       0.0       0.0       1.000E 01 0.0       0.0         -1.0
  -1
   6    TEST XS AND YIELD LIBRARY: FP
   6   10010 0.0       0.0       0.0       0.0       0.0       0.0          1.0
   6     0.0      0.0      5.00E 01 3.00E 01 0.0      0.0      0.0      0.0
   6   80160 0.0       0.0       0.0       0.0       0.0       0.0          1.0
   6     0.0      0.0      5.00E 01 7.00E 01 0.0      0.0      0.0      0.0
  -1
"""


origen_lib_all_decay = \
"""   1    TEST DECAY LIBRARY: ACTIVATION
   1   10010  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       9.998E 01 1.000E 00 1.000E 00
   1   20040  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       1.000E 02 1.000E 00 1.000E 00
  -1
   2    TEST DECAY LIBRARY: ACTINIDE
   2  902310  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  912350  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  912351  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  922350  1     1.386E 00 1.000E-02 7.000E-02 3.000E-02 2.000E-01 0.0
   2                3.000E-01 3.500E-01 0.0       7.204E-01 4.000E-12 3.000E-05
   2  922351  1     1.000E 00 0.0       0.0       0.0       0.0       1.000E 00
   2                0.0       0.0       0.0       7.204E-01 4.000E-12 3.000E-05
   2  932340  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  932350  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  932351  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
  -1
   3    TEST DECAY LIBRARY: FP
   3   10010  6     0.0       0.0       0.0       0.0       0.0       0.0
   3                0.0       0.0       0.0       9.998E 01 1.000E 00 1.000E 00
   3   80160  6     0.0       0.0       0.0       0.0       0.0       0.0
   3                0.0       0.0       0.0       9.976E 01 1.000E 00 1.000E 00
  -1
   4    TEST XS AND YIELD LIBRARY: ACTIVATION
  -1
   5    TEST XS AND YIELD LIBRARY: ACTINIDE
  -1
   6    TEST XS AND YIELD LIBRARY: FP
   6   10010 0.0       0.0       0.0       0.0       0.0       0.0          1.0
   6     0.0      0.0      5.00E 01 0.00E 01 0.0      0.0      0.0      0.0
   6   80160 0.0       0.0       0.0       0.0       0.0       0.0          1.0
   6     0.0      0.0      5.00E 01 0.00E 01 0.0      0.0      0.0      0.0
  -1
"""

origen_lib_all_xs = \
"""   1    TEST DECAY LIBRARY: ACTIVATION
   1   10010  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       0.0       1.000E 00 1.000E 00
   1   20040  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       0.0       1.000E 00 1.000E 00
   1   60130  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   1   70160  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   1   80160  6     0.0       0.0       0.0       0.0       0.0       0.0
   1                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
  -1
   2    TEST DECAY LIBRARY: ACTINIDE
   2  922330  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  922340  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  922341  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  922350  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  922360  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
   2  922361  6     0.0       0.0       0.0       0.0       0.0       0.0
   2                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
  -1
   3    TEST DECAY LIBRARY: FP
   3   80160  6     0.0       0.0       0.0       0.0       0.0       0.0
   3                0.0       0.0       0.0       0.0       4.000E-08 2.000E-04
  -1
   4    TEST XS AND YIELD LIBRARY: ACTIVATION
   4   80160 0.0       0.0       3.000E-01 7.000E-01 0.0       0.0          1.0
   4     0.0      0.0      1.00E 02 0.00E 01 0.0      0.0      0.0      0.0
  -1
   5    TEST XS AND YIELD LIBRARY: ACTINIDE
   5  922350 8.000E-01 1.200E 00 3.000E 00 4.000E 00 2.000E-01 8.000E-01   -1.0
  -1
   6    TEST XS AND YIELD LIBRARY: FP
   6   80160 0.0       0.0       3.000E-01 7.000E-01 0.0       0.0          1.0
   6     0.0      0.0      1.00E 02 0.00E 01 0.0      0.0      0.0      0.0
  -1
"""


# The following is the same as the above, except with the re-ordering due
# to using ORIGEN's isotope types and the additional f.p. precision
origen_lib_output = \
"""   1    TEST DECAY LIBRARY: ACTIVATION
   1  10010 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   1 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99844260E+01 1.00000000E+00
     1.00000000E+00
   1  10020 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   1 0.00000000E+00 0.00000000E+00 0.00000000E+00 1.55740000E-02 1.00000000E+00
     1.00000000E+00
   1  20040 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   1 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99998000E+01 1.00000000E+00
     1.00000000E+00
   1  70160 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   1 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 3.00000000E-08
     1.00000000E+00
   1  80160 1 3.46500000E-01 0.00000000E+00 1.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   1 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.97620600E+01 1.00000000E+00
     1.00000000E+00
  -1
   2    TEST DECAY LIBRARY: ACTINIDE
   2  20040 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99998000E+01 1.00000000E+00
     1.00000000E+00
   2 902310 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 2.01000000E+02 0.00000000E+00 4.00000000E-08
     2.00000000E-04
   2 902320 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99800000E+01 1.00000000E-12
     2.00000000E-06
   2 922330 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 4.00000000E-12
     3.00000000E-05
   2 922350 1 1.38600000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              1.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 2.00000000E+02 7.20400000E-01 4.00000000E-12
     3.00000000E-05
   2 922380 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 2.02000000E+02 9.92742000E+01 3.00000000E-12
     4.00000000E-05
   2 942390 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 6.00000000E-14
     5.00000000E-06
   2 942410 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 3.00000000E-12
     2.00000000E-04
   2 962450 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 2.00000000E-13
     4.00000000E-06
   2 982520 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 7.00000000E-13
     2.00000000E-05
  -1
   3    TEST DECAY LIBRARY: FP
  -1
   4    TEST XS AND YIELD LIBRARY: ACTIVATION
  -1
   5    TEST XS AND YIELD LIBRARY: ACTINIDE
   5 922350 0.00000000E+00 0.00000000E+00 0.00000000E+00 1.00000000E+02
            0.00000000E+00 0.00000000E+00 -1.0
   5 922380 0.00000000E+00 0.00000000E+00 0.00000000E+00 1.00000000E+01
            0.00000000E+00 0.00000000E+00 -1.0
  -1
   6    TEST XS AND YIELD LIBRARY: FP
  -1
"""

origen_lib_all_decay_output = \
"""   1    TEST DECAY LIBRARY: ACTIVATION
  -1
   2    TEST DECAY LIBRARY: ACTINIDE
   2  20040 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99998000E+01 1.00000000E+00
     1.00000000E+00
   2 902310 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 4.00000000E-08
     2.00000000E-04
   2 902320 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99800000E+01 1.00000000E-12
     2.00000000E-06
   2 912350 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 3.00000000E-08
     1.00000000E+00
   2 912351 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
     0.00000000E+00
   2 922330 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 4.00000000E-12
     3.00000000E-05
   2 922350 1 1.38600000E+00 1.00000000E-02 7.00000000E-02 3.00000000E-02
              2.00000000E-01 0.00000000E+00
   2 3.00000000E-01 3.50000000E-01 0.00000000E+00 7.20400000E-01 4.00000000E-12
     3.00000000E-05
   2 922351 1 1.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 1.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
     0.00000000E+00
   2 922380 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.92742000E+01 3.00000000E-12
     4.00000000E-05
   2 932340 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
     0.00000000E+00
   2 932350 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 2.00000000E-14
     3.00000000E-08
   2 932351 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
     0.00000000E+00
   2 942390 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 6.00000000E-14
     5.00000000E-06
   2 942410 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 3.00000000E-12
     2.00000000E-04
   2 962450 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 2.00000000E-13
     4.00000000E-06
   2 982520 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 7.00000000E-13
     2.00000000E-05
  -1
   3    TEST DECAY LIBRARY: FP
   3  10010 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99844260E+01 1.00000000E+00
     1.00000000E+00
   3  80160 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.97620600E+01 1.00000000E+00
     1.00000000E+00
  -1
   4    TEST XS AND YIELD LIBRARY: ACTIVATION
  -1
   5    TEST XS AND YIELD LIBRARY: ACTINIDE
  -1
   6    TEST XS AND YIELD LIBRARY: FP
   6  10010 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
            0.00000000E+00 0.00000000E+00 1.0
   6 0.00000000E+00 0.00000000E+00 5.00000000E+01 0.00000000E+00 0.00000000E+00
     0.00000000E+00 0.00000000E+00 0.00000000E+00
   6  80160 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
            0.00000000E+00 0.00000000E+00 1.0
   6 0.00000000E+00 0.00000000E+00 5.00000000E+01 0.00000000E+00 0.00000000E+00
     0.00000000E+00 0.00000000E+00 0.00000000E+00
  -1
"""

origen_lib_all_xs_output = \
"""   1    TEST DECAY LIBRARY: ACTIVATION
  -1
   2    TEST DECAY LIBRARY: ACTINIDE
   2 902320 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99800000E+01 1.00000000E-12
     2.00000000E-06
   2 922330 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 4.00000000E-12
     3.00000000E-05
   2 922340 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 5.40000000E-03 4.00000000E-12
     3.00000000E-05
   2 922341 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
     0.00000000E+00
   2 922350 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 7.20400000E-01 4.00000000E-12
     3.00000000E-05
   2 922360 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 4.00000000E-12
     3.00000000E-05
   2 922361 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
     0.00000000E+00
   2 922380 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.92742000E+01 3.00000000E-12
     4.00000000E-05
   2 942390 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 6.00000000E-14
     5.00000000E-06
   2 942410 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 3.00000000E-12
     2.00000000E-04
   2 962450 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 2.00000000E-13
     4.00000000E-06
   2 982520 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   2 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 7.00000000E-13
     2.00000000E-05
  -1
   3    TEST DECAY LIBRARY: FP
   3  10010 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99844260E+01 1.00000000E+00
     1.00000000E+00
   3  20040 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.99998000E+01 1.00000000E+00
     1.00000000E+00
   3  60130 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 1.10780000E+00 1.00000000E+00
     1.00000000E+00
   3  70160 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 3.00000000E-08
     1.00000000E+00
   3  80160 6 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
              0.00000000E+00 0.00000000E+00
   3 0.00000000E+00 0.00000000E+00 0.00000000E+00 9.97620600E+01 1.00000000E+00
     1.00000000E+00
  -1
   4    TEST XS AND YIELD LIBRARY: ACTIVATION
  -1
   5    TEST XS AND YIELD LIBRARY: ACTINIDE
   5 922350 8.00000000E-01 1.20000000E+00 3.00000000E+00 4.00000000E+00
            2.00000000E-01 8.00000000E-01 -1.0
  -1
   6    TEST XS AND YIELD LIBRARY: FP
   6  80160 0.00000000E+00 0.00000000E+00 3.00000000E-01 7.00000000E-01
            0.00000000E+00 0.00000000E+00 1.0
   6 0.00000000E+00 0.00000000E+00 1.00000000E+02 0.00000000E+00 0.00000000E+00
     0.00000000E+00 0.00000000E+00 0.00000000E+00
  -1
"""


def test_from_origen():
    # This tests the Depletion Library's ability to read from a file

    # The starting file is present in the origen_lib variable
    # So lets write it
    with open("test.lib", "w") as f:
        f.write(origen_lib)

    # Read it via from_origen
    xs_lib_ids = {"activation": 4, "actinide": 5, "fp": 6}
    decay_lib_ids = {"activation": 1, "actinide": 2, "fp": 3}
    lib = DepletionLibrary.from_origen("test.lib", "test.lib", xs_lib_ids,
                                       decay_lib_ids, new_name="TEST")

    # Now we have a DepletionLibrary
    # We will then compare to expectations
    assert lib.name == "TEST"
    assert np.array_equal(lib.neutron_group_structure, np.array([0., 20.]))
    # We test against this list here because from_origen adds in fissile
    # isotopes that yield data exist for.
    assert sorted(lib.isotopes) == \
        sorted(["H1", "H2", "He4", "O16", "N16", "Th231", "U235", "U238",
                "Th232", "U233", "Pu239", "Pu241", "Cm245", "Cf252"])
    assert sorted(lib.initial_isotopes) == \
        sorted(["H1", "H2", "He4", "O16", "N16", "Th231", "U235", "U238",
                "Th232", "U233", "Pu239", "Pu241", "Cm245", "Cf252"])

    # Check decay
    def check_decay(this, ref_t12, ref_t12_units, refQ, ref_prod_types,
                    ref_prod_brs, ref_prod_targets, ref_prod_yields):
        assert this.half_life == ref_t12
        assert this.half_life_units == ref_t12_units
        assert this.decay_energy == refQ
        assert sorted(this._products.keys()) == sorted(ref_prod_types)
        for i, type_ in enumerate(ref_prod_types):
            br, target, yield_ = this[type_]
            assert br == ref_prod_brs[i]
            assert target == ref_prod_targets[i]
            assert yield_ == ref_prod_yields[i]

    check_decay(lib.isotopes["H1"].decay, None, "s", 0., [], [], [], [])
    check_decay(lib.isotopes["H2"].decay, None, "s", 0., [], [], [], [])
    check_decay(lib.isotopes["Th231"].decay, None, "s", 201., [], [], [], [])
    check_decay(lib.isotopes["N16"].decay, None, "s", 0., [], [], [], [])
    check_decay(lib.isotopes["O16"].decay, 0.3465, "s", 0., ["ec/beta+"], [1.],
                [["N16"]], [[1.]])
    check_decay(lib.isotopes["U235"].decay, 1.386, "s", 200., ["alpha"],
                [1.], [["Th231", "He4"]], [[1., 1.]])
    check_decay(lib.isotopes["U238"].decay, None, "s", 202., [], [], [], [])

    # Now move on to comparing the xs
    def check_xs(this, ref_units, ref_groups, ref_types, ref_xss, ref_targets,
                 ref_yields, ref_q):
        assert this.xs_units == ref_units
        assert this.num_groups == ref_groups
        assert sorted(this.keys()) == ref_types
        for xs, targets, yields, q_val in this.values():
            assert xs == ref_xss
            assert targets == ref_targets
            assert yields == ref_yields
            assert q_val == ref_q

    # Now move on to comparing the xs
    xs = lib.isotopes["U235"].neutron_xs
    check_xs(xs, "b", 1, ["fission"], np.array([100.]), ["fission"],
             [1.0], 0.)
    xs = lib.isotopes["U238"].neutron_xs
    check_xs(xs, "b", 1, ["fission"], np.array([10.]), ["fission"],
             [1.0], 0.)

    # Finally compare the yields
    ref_channels = sorted(["H1", "O16"])
    # Check the ones we know have fission products
    nfy = lib.isotopes["U235"].neutron_fission_yield
    ref_vals = {k: 0.5 for k in ref_channels}
    for key in ref_vals.keys():
        assert nfy._products[key] == ref_vals[key]
    nfy = lib.isotopes["U238"].neutron_fission_yield
    ref_vals["H1"] = 0.3
    ref_vals["O16"] = 0.7
    for key in ref_vals.keys():
        assert nfy._products[key] == ref_vals[key]
    # Now check the rest
    for iso in ["Th232", "U233", "Pu239", "Pu241", "Cm245", "Cf252"]:
        nfy = lib.isotopes[iso].neutron_fission_yield
        ref_vals = {k: 0.0 for k in ref_channels}
        for key in ref_vals.keys():
            assert nfy._products[key] == ref_vals[key]


def test_from_origen_all_decay():
    # This tests the Depletion Library's ability to read from a file

    # The starting file is present in the origen_lib variable
    # So lets write it
    with open("test.lib", "w") as f:
        f.write(origen_lib_all_decay)

    # Read it via from_origen
    xs_lib_ids = {"activation": 4, "actinide": 5, "fp": 6}
    decay_lib_ids = {"activation": 1, "actinide": 2, "fp": 3}
    lib = DepletionLibrary.from_origen("test.lib", "test.lib", xs_lib_ids,
                                       decay_lib_ids, new_name="TEST")

    # Now we have a DepletionLibrary
    # We will then compare to expectations
    assert lib.name == "TEST"
    assert np.array_equal(lib.neutron_group_structure, np.array([0., 20.]))
    # We test against this list here because from_origen adds in fissile
    # isotopes that yield data exist for.
    manual_isos = ["H1", "He4", "O16", "Th231", "Pa235", "Pa235_m1", "U235",
                   "U235_m1", "Np234", "Np235", "Np235_m1"]
    yield_isos = ["Th232", "U233", "U238", "Pu239", "Pu241", "Cm245", "Cf252"]
    all_isos = sorted(manual_isos + yield_isos)
    assert sorted(lib.isotopes) == all_isos
    assert sorted(lib.initial_isotopes) == all_isos

    # Check decay
    def check_decay(this, ref_t12, ref_t12_units, refQ, ref_prod_types,
                    ref_prod_brs, ref_prod_targets, ref_prod_yields):
        assert this.half_life == ref_t12
        assert this.half_life_units == ref_t12_units
        assert this.decay_energy == refQ
        assert sorted(this._products.keys()) == sorted(ref_prod_types)
        for i, type_ in enumerate(ref_prod_types):
            br, target, yield_ = this[type_]
            assert abs(br - ref_prod_brs[i]) < 1e-15
            assert target == ref_prod_targets[i]
            np.testing.assert_allclose(yield_, ref_prod_yields[i], atol=1.e-15)

    ref_channels = sorted(["H1", "O16"])
    for iso_name in yield_isos:
        # Should be no neutron_xs, no decay, and I inputted 0 values for
        # yields
        iso = lib.isotopes[iso_name]
        assert iso.decay is None
        assert iso.neutron_xs is None
        # Do the yields
        nfy = iso.neutron_fission_yield
        ref_vals = {k: 0.0 for k in ref_channels}
        for key in ref_vals.keys():
            assert nfy._products[key] == ref_vals[key]

    for iso_name in manual_isos:
        # Should have manually set decay info, no neutron_xs, and
        # yield will exist for only U235
        iso = lib.isotopes[iso_name]
        assert iso.neutron_xs is None
        if iso_name not in ["U235", "U235_m1"]:
            check_decay(iso.decay, None, "s", 0., [], [], [], [])
        elif iso_name == "U235_m1":
            check_decay(iso.decay, 1.0, "s", 0., ["it"], [1.], [["U235"]],
                        [[1.]])
        elif iso_name == "U235":
            check_decay(iso.decay, 1.386, "s", 0.,
                        ["beta-", "ec/beta+", "alpha", "sf", "beta-,n"],
                        [0.05, 0.1, 0.2, 0.3, 0.35],
                        [["Np235", "Np235_m1"], ["Pa235", "Pa235_m1"],
                         ["Th231", "He4"], ["fission"], ["Np234"]],
                        [[0.8, 0.2], [0.7, 0.3], [1., 1.], [1.], [1.]])

        # Check the yields
        if iso_name != "U235":
            assert iso.neutron_fission_yield is None
        else:
            nfy = iso.neutron_fission_yield
            ref_vals = {k: 0.5 for k in ref_channels}
            for key in ref_vals.keys():
                assert nfy._products[key] == ref_vals[key]


def test_from_origen_all_xs():
    # This tests the Depletion Library's ability to read from a file

    # The starting file is present in the origen_lib variable
    # So lets write it
    with open("test.lib", "w") as f:
        f.write(origen_lib_all_xs)

    # Read it via from_origen
    xs_lib_ids = {"activation": 4, "actinide": 5, "fp": 6}
    decay_lib_ids = {"activation": 1, "actinide": 2, "fp": 3}
    lib = DepletionLibrary.from_origen("test.lib", "test.lib", xs_lib_ids,
                                       decay_lib_ids, new_name="TEST")

    # Now we have a DepletionLibrary
    # We will then compare to expectations
    assert lib.name == "TEST"
    assert np.array_equal(lib.neutron_group_structure, np.array([0., 20.]))
    # We test against this list here because from_origen adds in fissile
    # isotopes that yield data exist for.
    manual_isos = ["H1", "He4", "C13", "N16", "O16", "U233", "U234", "U234_m1",
                   "U235", "U236", "U236_m1"]
    yield_isos = ["Th232", "U238", "Pu239", "Pu241", "Cm245", "Cf252"]
    all_isos = sorted(manual_isos + yield_isos)
    assert sorted(lib.isotopes) == all_isos
    assert sorted(lib.initial_isotopes) == all_isos

    # Check decay
    def check_stable_decay(this):
        assert this.half_life is None
        assert this.half_life_units == "s"
        assert this.decay_constant == 0.
        assert this.decay_energy == 0.
        assert len(this._products) == 0

    def check_xs(this, ref_units, ref_groups, ref_type_data):
        assert this.xs_units == ref_units
        assert this.num_groups == ref_groups
        assert sorted(this.keys()) == sorted(ref_type_data.keys())
        for key, (xs, targets, yields, q_val) in this.items():
            ref_xs, ref_targets, ref_yields, ref_q = ref_type_data[key]
            ref_xs = np.array([ref_xs])
            assert xs == ref_xs
            assert targets == ref_targets
            assert yields == ref_yields
            assert q_val == ref_q

    ref_channels = sorted(["O16"])
    for iso_name in yield_isos:
        # Should be no neutron_xs, no decay, and I inputted 0 values for
        # yields
        iso = lib.isotopes[iso_name]
        assert iso.decay is None
        assert iso.neutron_xs is None
        # Do the yields
        nfy = iso.neutron_fission_yield
        ref_vals = {k: 0.0 for k in ref_channels}
        for key in ref_vals.keys():
            assert nfy._products[key] == ref_vals[key]

    for iso_name in manual_isos:
        # Should have manually set xs info, no decay, and
        # yield will exist for only U235
        iso = lib.isotopes[iso_name]
        check_stable_decay(iso.decay)
        if iso_name == "U235":
            check_xs(iso.neutron_xs, "b", 1,
                     {"(n,gamma)": (1., ["U236", "U236_m1"], [0.8, 0.2], 0.),
                      "(n,2n)": (2., ["U234", "U234_m1"], [0.6, 0.4], 0.),
                      "(n,3n)": (3., ["U233"], [1.], 0.),
                      "fission": (4., ["fission"], [1.0], 0.)})
        elif iso_name == "O16":
            check_xs(iso.neutron_xs, "b", 1,
                     {"(n,a)": (0.3, ["C13", "He4"], [1.0, 1.0], 0.),
                      "(n,p)": (0.7, ["N16", "H1"], [1.0, 1.0], 0.)})
        else:
            assert iso.neutron_xs is None

        nfy = iso.neutron_fission_yield
        if iso_name == "U235":
            ref_vals = {k: 1. for k in ref_channels}
            for key in ref_vals.keys():
                assert nfy._products[key] == ref_vals[key]
        elif iso_name == "U233":
            ref_vals = {k: 0.0 for k in ref_channels}
            for key in ref_vals.keys():
                assert nfy._products[key] == ref_vals[key]
        else:
            assert nfy is None


def test_to_origen():
    # This tests the Depletion Library's ability to write to an ORIGEN
    # file

    # The starting file is present in the origen_lib variable
    # So lets write it
    for in_lib, out_lib in zip([origen_lib, origen_lib_all_decay,
                                origen_lib_all_xs],
                               [origen_lib_output, origen_lib_all_decay_output,
                                origen_lib_all_xs_output]):
        with open("test_in.lib", "w") as f:
            f.write(in_lib)

        # Read it via from_origen
        xs_lib_ids = {"activation": 4, "actinide": 5, "fp": 6}
        decay_lib_ids = {"activation": 1, "actinide": 2, "fp": 3}
        lib = DepletionLibrary.from_origen("test_in.lib", "test_in.lib",
                                           xs_lib_ids, decay_lib_ids,
                                           new_name="TEST")
        lib.set_isotope_indices()
        if in_lib != origen_lib:
            # Force the library to figure out isotope types on its own
            # This is only needed because of the metastables which dont
            # really exist in the _all_decay and _all_xs libs
            isotope_types = Origen22Depletion.assign_isotope_types(lib)
        else:
            isotope_types = ORIGEN_ISO_DECAY_TYPES
        to_origen(lib, isotope_types, "test_out.lib")

        output = []
        with open("test_out.lib", "r") as f:
            lines = f.readlines()
            for line in lines:
                # Remove any trailing whitespace
                line = line.rstrip()
                output.append(line)

        output = "\n".join(output) + "\n"

        assert out_lib == output


def test_to_origen_2g(depletion_lib, depletion_lib_2g):
    # This tests the Depletion Library's ability to write a 2g library
    # to a file
    isotope_types = ORIGEN_ISO_DECAY_TYPES

    # First create the 1g file
    depletion_lib.set_isotope_indices()
    to_origen(depletion_lib, isotope_types, "test_1g.lib")

    # Now the 2g file, use equal flux weighting as that gives us
    # the same results as the 1g lib b design of the problem
    depletion_lib_2g.set_isotope_indices()
    to_origen(depletion_lib_2g, isotope_types, "test_2g.lib",
              flux=np.array([1., 2.]))

    # Now read each file and compare the library
    output_1g = []
    with open("test_1g.lib", "r") as f:
        lines = f.readlines()
        for line in lines:
            # Remove any trailing whitespace
            line = line.rstrip()
            output_1g.append(line)
    output_1g = "\n".join(output_1g) + "\n"

    output_2g = []
    with open("test_2g.lib", "r") as f:
        lines = f.readlines()
        for line in lines:
            # Remove any trailing whitespace
            line = line.rstrip()
            output_2g.append(line)
    output_2g = "\n".join(output_2g) + "\n"

    assert output_1g == output_2g


def test_to_hdf5():
    # This tests the Depletion Library's ability to read and write to
    # an HDF5

    # We will take the origen library, read it, write it, read it, and
    # compare it

    # The starting file is present in the origen_lib variable
    # So lets write it
    with open("test.lib", "w") as f:
        f.write(origen_lib)

    # Read it via from_origen
    xs_lib_ids = {"activation": 4, "actinide": 5, "fp": 6}
    decay_lib_ids = {"activation": 1, "actinide": 2, "fp": 3}
    orig_lib = DepletionLibrary.from_origen("test.lib", "test.lib", xs_lib_ids,
                                            decay_lib_ids, new_name="TEST")

    orig_lib.to_hdf5("test.h5")
    lib = DepletionLibrary.from_hdf5("test.h5", "TEST")
    lib.set_isotope_indices()

    # Now we have a DepletionLibrary
    # We will then compare to expectations
    assert lib.name == "TEST"
    assert np.array_equal(lib.neutron_group_structure, np.array([0., 20.]))
    # We test against this list here because from_origen adds in fissile
    # isotopes that yield data exist for.
    assert sorted(lib.isotopes) == \
        sorted(["H1", "H2", "He4", "O16", "N16", "Th231", "U235", "U238",
                "Th232", "U233", "Pu239", "Pu241", "Cm245", "Cf252"])
    assert sorted(lib.initial_isotopes) == \
        sorted(["H1", "H2", "He4", "O16", "N16", "Th231", "U235", "U238",
                "Th232", "U233", "Pu239", "Pu241", "Cm245", "Cf252"])

    # Check the decay
    def check_decay(this, ref_t12, ref_t12_units, refQ, ref_prod_types,
                    ref_prod_brs, ref_prod_targets, ref_prod_yields):
        assert this.half_life == ref_t12
        assert this.half_life_units == ref_t12_units
        assert this.decay_energy == refQ
        assert sorted(this._products.keys()) == sorted(ref_prod_types)
        for i, type_ in enumerate(ref_prod_types):
            br, target, yield_ = this[type_]
            assert br == ref_prod_brs[i]
            assert target == ref_prod_targets[i]
            assert yield_ == ref_prod_yields[i]

    check_decay(lib.isotopes["H1"].decay, None, "s", 0., [], [], [], [])
    check_decay(lib.isotopes["H2"].decay, None, "s", 0., [], [], [], [])
    check_decay(lib.isotopes["Th231"].decay, None, "s", 201., [], [], [], [])
    check_decay(lib.isotopes["N16"].decay, None, "s", 0., [], [], [], [])
    check_decay(lib.isotopes["O16"].decay, 0.3465, "s", 0., ["ec/beta+"], [1.],
                [["N16"]], [[1.]])
    check_decay(lib.isotopes["U235"].decay, 1.386, "s", 200., ["alpha"],
                [1.], [["Th231", "He4"]], [[1., 1.]])
    check_decay(lib.isotopes["U238"].decay, None, "s", 202., [], [], [], [])

    # Now move on to comparing the xs
    def check_xs(this, ref_units, ref_groups, ref_types, ref_xss, ref_targets,
                 ref_yields, ref_q):
        assert this.xs_units == ref_units
        assert this.num_groups == ref_groups
        assert sorted(this.keys()) == ref_types
        for xs, targets, yields, q_val in this.values():
            assert xs == ref_xss
            assert targets == ref_targets
            assert yields == ref_yields
            assert q_val == ref_q

    xs = lib.isotopes["U235"].neutron_xs
    check_xs(xs, "b", 1, ["fission"], np.array([100.]), ["fission"],
             [1.0], 0.)
    xs = lib.isotopes["U238"].neutron_xs
    check_xs(xs, "b", 1, ["fission"], np.array([10.]), ["fission"],
             [1.0], 0.)

    # Finally compare the yields
    ref_channels = sorted(["H1", "O16"])
    # Check the ones we know have fission products
    nfy = lib.isotopes["U235"].neutron_fission_yield
    ref_vals = {k: 0.5 for k in ref_channels}
    for key in ref_vals.keys():
        assert nfy._products[key] == ref_vals[key]
    nfy = lib.isotopes["U238"].neutron_fission_yield
    ref_vals["H1"] = 0.3
    ref_vals["O16"] = 0.7
    for key in ref_vals.keys():
        assert nfy._products[key] == ref_vals[key]
    # Now check the rest
    for iso in ["Th232", "U233", "Pu239", "Pu241", "Cm245", "Cf252"]:
        nfy = lib.isotopes[iso].neutron_fission_yield
        ref_vals = {k: 0.0 for k in ref_channels}
        for key in ref_vals.keys():
            assert nfy._products[key] == ref_vals[key]


def test_A_matrix(depletion_lib_2g):
    # The matrix will be based on depletion_lib_2g
    depletion_lib_2g.isotope_indices = OrderedDict()
    depletion_lib_2g._isotopes_ordered = False

    # Compute our reference decay matrix and A
    # will use a flux vector of [1., 2.]
    flux = np.array([1., 2.])
    ref_order = ["H1", "He4", "N16", "O16", "Th231", "U235", "U238"]
    D = np.zeros((len(ref_order), len(ref_order)))
    D[6, 6] = 0.  # U238, stable
    D[5, 5] = -0.5  # U235 alphas to only Th231
    D[4, 5] = 0.5  # The Th231 part
    D[1, 5] = 0.5  # The alpha-particle from U235
    D[4, 4] = 0.  # Th231, stable
    D[3, 3] = -2.  # O16 decays to N16
    D[2, 3] = 2.
    D[2, 2] = 0.  # N16, stable
    D[1, 1] = 0.  # He4, stable
    D[0, 0] = 0.  # H1, stable
    A = np.copy(D)
    # U238 fissions to H1, O16 w/ [10, 10] xs and no xs rxns produce it
    A[6, 6] += -(np.dot([5., 12.5], flux)) * 1.E-24
    # U235 fissions to H1, N16 and no xs rxns produce it
    A[5, 5] += -(np.dot([50., 125.], flux)) * 1.E-24
    # Thorium has no xs and no xs rxns produce it
    # O16 has no xs, but is a U235 fp with a yield of 0.5
    A[3, 5] += 0.5 * (np.dot([50., 125.], flux)) * 1.E-24
    # N16 has no xs, but is a U238 fp with a yield of 1.0 each (after scaling)
    A[2, 6] += 1.0 * (np.dot([5., 12.5], flux)) * 1.E-24
    # H1 has no xs, but is a U235 (0.5 yield) and U238 (1.0 yield) fp
    A[0, 6] += 1.0 * (np.dot([5., 12.5], flux)) * 1.E-24
    A[0, 5] += 0.5 * (np.dot([50., 125.], flux)) * 1.E-24

    test_decay_matrix = depletion_lib_2g.build_decay_matrix()

    assert np.array_equal(ref_order,
                          [k for k in depletion_lib_2g.isotope_indices.keys()])
    np.testing.assert_allclose(D, test_decay_matrix, rtol=1.e-14)

    # Ok, now we can calculate A
    # Some scipy versions are out of date with numpy and will result in
    # a deprecation warning about using the matrix subclass
    # This is in the todense function; we can ignore that warning since
    # we are aware of it and it has no effect
    # we could do a try/except but to catch a warning (vice error)
    # will require extra code that doesn't seem worth it.
    testA = depletion_lib_2g.build_depletion_matrix(flux).todense()
    np.testing.assert_allclose(A, testA, rtol=1.e-14)


def test_clone(depletion_lib_2g):
    # This will make a clone of depletion_lib_2g to verify that we get the
    # expected shallow vs deep copy behavior
    decay_matrix = depletion_lib_2g.build_decay_matrix()
    the_clone = depletion_lib_2g.clone(new_name="new name")
    clone_decay_matrix = the_clone.build_decay_matrix()

    # Make sure this is a different object, as one would expect
    assert the_clone is not depletion_lib_2g

    # Check the decay matrix
    assert np.all(decay_matrix == clone_decay_matrix)

    # Check the metadata
    assert the_clone.name != depletion_lib_2g.name
    assert the_clone.name == "new name"
    attribs = ["neutron_group_structure", "_isotopes_ordered",
        "isotope_indices"]
    # Now make sure the items we expect to be deep copied are not references
    # to the same thing but the values are the same
    for attrib in attribs:
        test = getattr(the_clone, attrib)
        ref = getattr(depletion_lib_2g, attrib)
        if attrib != "_isotopes_ordered":
            assert test is not ref
        if attrib in ["_isotopes_ordered", "isotope_indices"]:
            assert test == ref
        else:
            np.testing.assert_array_equal(test, ref)

    # Now dive in to the isotopic data and make sure that was cloned as
    # expected
    for iso_name in the_clone.isotopes.keys():
        test = the_clone.isotopes[iso_name]
        ref = depletion_lib_2g.isotopes[iso_name]

        # Check the shallow copies
        attribs = ["name", "atomic_mass", "neutron_fission_yield", "decay"]
        for attrib in attribs:
            test_a = getattr(test, attrib)
            ref_a = getattr(ref, attrib)
            assert test_a is ref_a

        # And the deep copies
        attribs = ["neutron_xs", "removal"]
        for attrib in attribs:
            test_a = getattr(test, attrib)
            ref_a = getattr(ref, attrib)

            if test_a is not None:
                # We're not going to check the actual values within the
                # neutron_xs and removal objects as that is tested thoroughly
                # elsewhere, and we are using Python built-ins for this.
                assert test_a is not ref_a
            else:
                # Then test_a is None, so make sure ref_a is
                assert ref_a is None



