from __future__ import annotations

import re

ANONYMOUS_USER_AGENT = "OpenLitReview/0.1"
ANONYMOUS_JSON_HEADERS = {
    "User-Agent": ANONYMOUS_USER_AGENT,
    "Accept": "application/json",
}

_EMAIL_PATTERN = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CN_ID_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def reject_contact_identifiers(value: str) -> None:
    """Stop obvious contact or identity values from becoming scholarly search queries."""
    if _EMAIL_PATTERN.search(value):
        raise ValueError("Academic search terms must not contain an email address")
    if _PHONE_PATTERN.search(value):
        raise ValueError("Academic search terms must not contain a mobile phone number")
    if _CN_ID_PATTERN.search(value):
        raise ValueError("Academic search terms must not contain a national ID number")
