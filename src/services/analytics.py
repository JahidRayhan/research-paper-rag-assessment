"""Topic extraction for /analytics/popular (stop-word filtered unigrams + bigrams)."""
import re

STOP = set("""a an the and or of to in on for with by from at as is are was were be been being this that these
those it its what which who whom how why when where does do did can could should would will shall may might
about across into between among over under than then there their they them we our you your i me my not no yes
use used using paper papers provided discuss describe explain compare list summarize summarise major main
different various also such each any all some more most other""".split())


def extract_terms(text: str) -> list[str]:
    tokens = [t for t in re.findall(r"[a-z][a-z\-]{2,}", text.lower()) if t not in STOP]
    terms = list(tokens)
    terms += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return terms
