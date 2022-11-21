import numpy as np

# The MCNP manual states that the type of data card must begin within
# the first five columns of the input file.
# This constant stores this information
CARD_TYPE_SIZE = 5

# MCNP maximum line length
MAX_LINE_LEN = 80

# The message block key per Section 2.4 (NOTE this constant must be
# a lowercase for this script)
MESSAGE_KEY = "message:"

# The MCNP comment and line continuation characters
NEWLINE_COMMENT = "c"
COMMENT = "$"
CONTINUATION = "&"

# Data block vertical format delineator
VERTICAL = "#"

# The confidence interval to use when rejecting or accepting a test position
# in a critical search
CI_995 = 2.807034   # Scale sigma by this for 99.5% CI
CI_95 = 1.960       # Scale sigma by this for 95% CI

# Per Table 3-2, the maximum cell and material ids are 99,999,999
UNIV_MAX_ID = 99999999
CELL_MAX_ID = 99999999
MATL_MAX_ID = 99999999
TALLY_MAX_ID = 99999999
SURF_MAX_ID = 99999999
TRANSFORM_MAX_ID = 999
MATL_VOID = 0
ROOT_UNIV = 0
IRREG_LAT_ARRAY = 0
USE_LAT_IRREG = -2
USE_LAT_MAT = -1
MAX_NUM_TALLIES = 9999
MAX_NUM_TRANSFORM = 999
MAX_SURF_NUM_TRANSFORM = 999
MAX_CELL_NUM_TRANSFORM = 999

DEFAULT_DISPLACEMENT = np.zeros(3)
DEFAULT_ROT_MAT = np.eye(3)

VALID_FILL_TYPES = [None, "single", "array"]

MATL_TUPLE_CELLS, MATL_TUPLE_ZAMS_NLIBS, MATL_TUPLE_FRACTIONS, \
    MATL_THERMAL = (0, 1, 2, 3)

# The final letter(s) of cases run with MCNP6 from Chap 7-5 of the users manual
MCNP_OUT_SUFFIX = ["m", "e", "d", "o", "s", "1", "h", "linkout",
                   "r", "p", "msht"]
# And other filetype names one would get from mcnp5 or 6
# (Sec 7.1.3 of MCNP6 manual)
MCNP_OUT_NAMES = ["com", "comout", "histp", "ksental", "linkout", "mctal",
                  "mdata", "meshtal", "outp", "plotm", "plotm.ps", "ptrac",
                  "runtpe", "srctp", "wssa", "wwone", "wwout", "wxxa"]
# The suffices to use when determining if a fast-forward run is possible
MCNP_FF_SUFFIX = ["m", "o"]

# The following cards result in an error message from ADDER
ERROR_CARDS = ["burn", "embeb", "embed", "embee", "embem", "embtb", "embtm",
               "notrn", "talnp", "read", "continue"]

# The following are the tally cards per the unnumbered table in
# Section 3.3.5
TALLY_CARDS = ["f", "fip", "fir", "fic", "fc", "e", "t", "c", "fq", "fm", "tm",
               "de", "df", "em", "cm", "cf", "sf", "fs", "sd", "fu",
               "ft", "tf", "notrn", "pert", "kpert", "ksen", "fmesh"]
TMESH_CARDS = ["tmesh", "endmd"]

# The following are output control cards that we want to look for
OUTPUT_CARDS = ["print", "talnp", "prdmp", "ptrac", "mplot", "histp", "dbcn"]

# The following are the simulation setting cards we need to look for
SIM_SETTING_CARDS = ["kcode", "sdef", "si", "sp", "sb", "ds", "sc", "ssw",
                     "ssr", "ksrc", "kopts", "hsrc", "source", "srcdx"]

# The cards which start with "m" that are not materials
NOT_MAT_CARDS = ["mesh", "mgopt", "mode", "mphys", "mplot", "mx", "mp",
                 "mshmf1", "mshmf2", "mshmf3", "mshmf4", "mshmf5", "mshmf6",
                 "mshmf7", "mshmf8", "mshmf9"]

# The tally id to use for ADDER-specific tallies
ADDER_TALLY_ID = TALLY_MAX_ID - 10 + 1

# The print tables that ADDER requires in MCNP output
PRINT_TABLES = [60, 128, 130]

MCNP_RXN_TALLY_IDS = {
    "(n,gamma)": 102, "(n,2n)": 16, "(n,3n)": 17, "(n,4n)": 37, "(n,a)": 107,
    "fission": -6, "(n,p)": 103, "(n,d)": 104, "(n,t)": 105, "(n,np)": 28}

# The following is a list of CELL card keywords:
# Note that the starred versions must always come before the unstarred
CELL_KW = ["*trcl", "imp:", "vol", "pwt", "ext:", "fcl:", "wwn:", "dxc:",
           "nonu", "pd", "tmp", "tmp1", "tmp2", "tmp3", "u", "trcl", "lat",
           "fill", "*fill", "elpt:", "cosy", "bflcl", "unc:"]
LIKE_CELL_KW = CELL_KW + ["mat", "rho"]

# These are cards which can also be cell keywords; those in this list will
# be moved from the user's input to cell card keywords
CARD_TO_CELL = ["vol", "pwt", "u", "nonu", "lat", "fill", "*fill", "cosy",
                "bflcl", "imp:", "ext:", "fcl:", "elpt:", "unc:", "dxc", "pd",
                "tmp", "wwn"]


# The allowed particle designators from Table 2-2
PARTICLE_TYPES = ["n", "p", "e", "|", "q", "u", "v", "f", "h", "l", "+", "-",
                  "x", "y", "o", "!", "<", ">", "g", "/", "z", "k", "%", "^",
                  "b", "_", "~", "c", "w", "@", "d", "t", "s", "a", "*", "?",
                  "#"]

# Material keywords
MAT_KW = ["gas", "estep", "hstep", "nlib", "plib", "pnlib", "elib", "hlib",
          "cond"]

CELL_DENSITY_FMT = "{:.13E}"
