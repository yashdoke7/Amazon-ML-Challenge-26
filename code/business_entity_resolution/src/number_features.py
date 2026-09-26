"""Optional learned first-address-number features for a compatible matcher."""

import re

from features import FEATURE_NAMES, pair_features

NUMBER = re.compile(r"\d+")
NUMBER_FEATURE_NAMES = FEATURE_NAMES + [
    "first_number_equal", "first_number_distance", "first_number_near", "first_numbers_present"]


def first_number(value):
    match = NUMBER.search(value or "")
    return int(match.group()) if match else -1


def number_pair_features(q_name, t_name, q_address, t_address, source):
    qnum = first_number(q_address)
    tnum = first_number(t_address)
    present = qnum >= 0 and tnum >= 0
    distance = abs(qnum - tnum) if present else -1
    return (*pair_features(q_name, t_name, q_address, t_address, source),
            int(present and qnum == tnum), min(distance, 1000) if present else -1,
            int(present and 0 < distance <= 10), int(present))
