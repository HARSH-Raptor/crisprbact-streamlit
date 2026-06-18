_COMP_TABLE = str.maketrans("ATGC", "TACG")


def rev_comp(seq):
    return seq.translate(_COMP_TABLE)[::-1]


class NoRecordsException(Exception):
    """No Record found in the sequence file"""

    pass
