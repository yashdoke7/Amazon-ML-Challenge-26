"""Local pair features used by the challenge matcher. No remote calls or data."""

from functools import lru_cache
import re
import unicodedata

import regex
from rapidfuzz import fuzz


TOKEN_RX = regex.compile(r"[\p{L}\p{M}\p{N}]+")
NUMBER_RX = re.compile(r"\d+")
LEGAL_RX = re.compile(r"\b(?:inc|llc|ltd|limited|private|pvt|corp|corporation|llp|co|company|sas|sarl|sa|eurl)\b")


@lru_cache(maxsize=1_000_000)
def normalize(value):
    return " ".join(TOKEN_RX.findall(unicodedata.normalize("NFKC", value or "").casefold()))


@lru_cache(maxsize=1_000_000)
def core(value):
    return " ".join(LEGAL_RX.sub(" ", normalize(value)).split())


@lru_cache(maxsize=1_000_000)
def fold(value):
    return "".join(c for c in unicodedata.normalize("NFKD", core(value)) if not unicodedata.combining(c))


@lru_cache(maxsize=1_000_000)
def parts(value):
    return frozenset(x for x in normalize(value).split() if len(x) >= 2)


@lru_cache(maxsize=1_000_000)
def numbers(value):
    return frozenset(x.lstrip("0") or "0" for x in NUMBER_RX.findall(value or ""))


FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "core_ratio", "core_token_sort",
    "accent_core_ratio", "core_exact", "name_exact", "address_ratio", "address_token_sort",
    "address_token_set", "name_common_tokens", "name_containment", "address_common_tokens",
    "address_containment", "common_address_numbers", "disjoint_address_numbers",
    "target_has_address", "query_name_len", "target_name_len", "query_address_len",
    "target_address_len", "source3",
]


def pair_features(q_name, t_name, q_addr, t_addr, source):
    qn, tn = normalize(q_name), normalize(t_name)
    qc, tc = core(q_name), core(t_name)
    qa, ta = normalize(q_addr), normalize(t_addr)
    qnt, tnt = parts(q_name), parts(t_name)
    qat, tat = parts(q_addr), parts(t_addr)
    qnums, tnums = numbers(q_addr), numbers(t_addr)
    name_common = len(qnt & tnt)
    addr_common = len(qat & tat)
    common_nums = len(qnums & tnums)
    return (
        fuzz.ratio(qn, tn), fuzz.token_sort_ratio(qn, tn), fuzz.token_set_ratio(qn, tn),
        fuzz.ratio(qc, tc), fuzz.token_sort_ratio(qc, tc), fuzz.ratio(fold(q_name), fold(t_name)),
        int(bool(qc) and qc == tc), int(bool(qn) and qn == tn),
        fuzz.ratio(qa, ta), fuzz.token_sort_ratio(qa, ta), fuzz.token_set_ratio(qa, ta),
        name_common, name_common / max(1, min(len(qnt), len(tnt))),
        addr_common, addr_common / max(1, min(len(qat), len(tat))),
        common_nums, int(bool(qnums and tnums) and common_nums == 0),
        int(bool(t_addr)), len(qn), len(tn), len(qa), len(ta),
        int(source == "S3"),
    )
