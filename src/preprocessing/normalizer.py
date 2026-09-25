"""Business Name, Address, and Country Normalization Engine.

Provides multi-representation text cleaning, unicode decomposition, legal suffix
stripping, address abbreviation expansion, and open-set country normalization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

# Controlled corporate legal suffixes (sorted by length descending for greedy matching)
LEGAL_SUFFIXES = [
    "private limited",
    "limited liability company",
    "pvt ltd",
    "pvt. ltd.",
    "pte ltd",
    "pty ltd",
    "corp.",
    "corp",
    "corporation",
    "incorporated",
    "inc.",
    "inc",
    "ltd.",
    "ltd",
    "limited",
    "llc.",
    "llc",
    "llp.",
    "llp",
    "pllc",
    "gmbh",
    "s.a.s.",
    "sas",
    "s.a.",
    "sa",
    "s.a.r.l.",
    "sarl",
    "co.",
    "co",
    "company",
]

# Canonical address token mapping
ADDRESS_EXPANSIONS = {
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "ave.": "avenue", "av": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "ct": "court", "ct.": "court",
    "pl": "place", "pl.": "place",
    "sq": "square", "sq.": "square",
    "ste": "suite", "ste.": "suite",
    "apt": "apartment", "apt.": "apartment",
    "fl": "floor", "fl.": "floor", "flr": "floor",
    "bldg": "building", "bldg.": "building",
    "hwy": "highway", "hwy.": "highway",
    "pkwy": "parkway", "pkwy.": "parkway",
    "terr": "terrace", "terr.": "terrace",
    "cir": "circle", "cir.": "circle",
    "n": "north", "n.": "north",
    "s": "south", "s.": "south",
    "e": "east", "e.": "east",
    "w": "west", "w.": "west",
    "ne": "northeast", "ne.": "northeast",
    "nw": "northwest", "nw.": "northwest",
    "se": "southeast", "se.": "southeast",
    "sw": "southwest", "sw.": "southwest",
}

# Country aliases mapping (open-set safe: unmapped countries retain cleaned string)
COUNTRY_ALIASES = {
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "us": "US",
    "in": "INDIA",
    "ind": "INDIA",
    "india": "INDIA",
    "bharat": "INDIA",
    "fr": "FRANCE",
    "fra": "FRANCE",
    "france": "FRANCE",
    "french republic": "FRANCE",
    "uk": "UNITED KINGDOM",
    "united kingdom": "UNITED KINGDOM",
    "great britain": "UNITED KINGDOM",
    "gb": "UNITED KINGDOM",
    "gbr": "UNITED KINGDOM",
    "de": "GERMANY",
    "deu": "GERMANY",
    "germany": "GERMANY",
    "deutschland": "GERMANY",
}

# Compiled regex patterns
RE_UNICODE_ACCENTS = re.compile(r'[\u0300-\u036f]')
RE_HTML_AMP = re.compile(r'&amp;', re.IGNORECASE)
RE_AMP = re.compile(r'&')
RE_AT = re.compile(r'@')
RE_PUNCT = re.compile(r'[^\w\s]')
RE_WHITESPACE = re.compile(r'\s+')
RE_US_ZIP = re.compile(r'\b\d{5}(?:-\d{4})?\b')
RE_INDIA_PIN = re.compile(r'\b\d{6}\b')
RE_DIGITS = re.compile(r'\b\d+\b')


def unicode_clean(text: str) -> str:
    """Decomposes unicode accents, normalizes quotes, and strips invalid chars."""
    if text is None:
        return ""
    s = str(text)
    # Replace HTML entity
    s = RE_HTML_AMP.sub(" and ", s)
    s = RE_AMP.sub(" and ", s)
    s = RE_AT.sub(" at ", s)
    # Normalize unicode (NFKD decomposes accented glyphs)
    s = unicodedata.normalize('NFKD', s)
    s = RE_UNICODE_ACCENTS.sub('', s)
    # Normalize quotes and dashes
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return s


def normalize_business_name(name: str) -> str:
    """Standardizes business name via lowercasing, unicode cleaning, and punctuation removal."""
    if not name:
        return ""
    s = unicode_clean(name).lower()
    s = RE_PUNCT.sub(' ', s)
    s = RE_WHITESPACE.sub(' ', s).strip()
    return s


def strip_legal_suffix(clean_name: str) -> str:
    """Removes trailing corporate legal entity designations from normalized name."""
    if not clean_name:
        return ""
    tokens = clean_name.split()
    if len(tokens) <= 1:
        return clean_name

    text = " " + clean_name + " "
    for suffix in LEGAL_SUFFIXES:
        s_clean = suffix.replace(".", "")
        pat = " " + s_clean + " "
        if text.endswith(pat):
            text = text[:-len(pat)].strip()
            return text
    return clean_name


def normalize_address(address: str) -> str:
    """Standardizes address by expanding abbreviations and normalizing tokens."""
    if not address:
        return ""
    s = unicode_clean(address).lower()
    # Remove commas, periods, hashes but keep tokens
    s = RE_PUNCT.sub(' ', s)
    tokens = s.split()
    expanded_tokens = [ADDRESS_EXPANSIONS.get(t, t) for t in tokens]
    return " ".join(expanded_tokens).strip()


def extract_postal_code(address: str) -> str:
    """Extracts 5-digit US zip or 6-digit India PIN code if present."""
    if not address:
        return ""
    # Check 6-digit PIN first
    m_in = RE_INDIA_PIN.search(address)
    if m_in:
        return m_in.group(0)
    # Check 5-digit US Zip
    m_us = RE_US_ZIP.search(address)
    if m_us:
        return m_us.group(0).split("-")[0]
    return ""


def extract_address_digits(address: str) -> str:
    """Extracts all standalone digit sequences from address (e.g. house/street numbers)."""
    if not address:
        return ""
    digits = RE_DIGITS.findall(address)
    return " ".join(digits)


def normalize_country(country: str) -> str:
    """Normalizes country names with open-set compatibility."""
    if not country:
        return "UNKNOWN"
    c_clean = unicode_clean(country).strip().lower()
    c_clean = RE_PUNCT.sub('', c_clean)
    c_clean = RE_WHITESPACE.sub(' ', c_clean).strip()
    
    # Check known aliases
    if c_clean in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[c_clean]
    # Fallback for open-set countries (e.g. France, Germany, Japan)
    return c_clean.upper() if c_clean else "UNKNOWN"


def transform_record(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
) -> Dict[str, str]:
    """Generates all normalized representations for a single business entity record."""
    orig_name = str(business_name) if business_name is not None else ""
    orig_addr = str(business_address) if business_address is not None else ""
    orig_cntry = str(country) if country is not None else ""

    norm_name = normalize_business_name(orig_name)
    suffix_stripped = strip_legal_suffix(norm_name)
    norm_addr = normalize_address(orig_addr)
    postal = extract_postal_code(orig_addr)
    addr_digits = extract_address_digits(orig_addr)
    norm_country = normalize_country(orig_cntry)

    return {
        "entity_id": str(entity_id).strip(),
        "original_name": orig_name.strip(),
        "normalized_name": norm_name,
        "clean_name_no_suffix": suffix_stripped,
        "original_address": orig_addr.strip(),
        "normalized_address": norm_addr,
        "postal_code": postal,
        "address_digits": addr_digits,
        "original_country": orig_cntry.strip(),
        "normalized_country": norm_country,
    }
