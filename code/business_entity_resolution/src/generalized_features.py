"""Optional script-agnostic comparison features for a measured matcher probe.

ASCII transliteration is a general text transform; it does not look up business
identities. Keep the original-script features as well so transliteration loss
cannot hide distinguishing characters from the model.
"""

from functools import lru_cache

from anyascii import anyascii
from rapidfuzz import fuzz

from features import normalize
from number_features import NUMBER_FEATURE_NAMES, number_pair_features


GENERALIZED_FEATURE_NAMES = NUMBER_FEATURE_NAMES + [
    "ascii_name_ratio", "ascii_name_token_sort", "ascii_name_token_set",
    "ascii_address_ratio", "ascii_address_token_set", "ascii_name_containment",
    "name_has_nonascii", "address_has_nonascii",
]


@lru_cache(maxsize=250_000)
def ascii_normalize(value):
    return normalize(anyascii(value))


def generalized_pair_features(q_name, t_name, q_addr, t_addr, source):
    q_name, t_name = q_name or "", t_name or ""
    q_addr, t_addr = q_addr or "", t_addr or ""
    base = number_pair_features(q_name, t_name, q_addr, t_addr, source)
    nonascii_name = any(ord(c) > 127 for c in q_name + t_name)
    nonascii_addr = any(ord(c) > 127 for c in q_addr + t_addr)
    if nonascii_name:
        qn, tn = ascii_normalize(q_name), ascii_normalize(t_name)
        name_ratio = fuzz.ratio(qn, tn)
        name_sort = fuzz.token_sort_ratio(qn, tn)
        name_set = fuzz.token_set_ratio(qn, tn)
        qt, tt = set(qn.split()), set(tn.split())
        containment = len(qt & tt) / max(1, min(len(qt), len(tt)))
    else:
        name_ratio, name_sort, name_set = base[0], base[1], base[2]
        containment = base[12]
    if nonascii_addr:
        qa, ta = ascii_normalize(q_addr), ascii_normalize(t_addr)
        address_ratio = fuzz.ratio(qa, ta)
        address_set = fuzz.token_set_ratio(qa, ta)
    else:
        address_ratio, address_set = base[8], base[10]
    return (*base, name_ratio, name_sort, name_set, address_ratio,
            address_set, containment, int(nonascii_name), int(nonascii_addr))
