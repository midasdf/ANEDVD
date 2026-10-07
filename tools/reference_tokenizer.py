#!/usr/bin/env python3
"""Independent GPT-2 byte-level BPE reference for the tiny fixture vocab.

Implements the published GPT-2 algorithm (byte encoder + regex pre-tokenizer +
rank-ordered merges) from scratch, using Python's `re` with an ASCII-exact
transcription of the GPT-2 pattern.  Used to check `src/tokenizer.zig` against
a second implementation and to print the expected ids hard-coded in its tests.

Usage: tools/reference_tokenizer.py
"""

import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from make_fixtures import MERGES, build_vocab, B2U  # noqa: E402

VOCAB = {t: i for i, t in enumerate(build_vocab())}

# GPT-2 pattern, with \p{L}/\p{N} expanded for the ASCII subset (the fixture
# test strings are ASCII except the UTF-8 ones, which are checked via the
# letter/number approximations documented in src/tokenizer.zig).
PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+(?!\S)|\s+")


def byte_encode(text):
    return "".join(chr(B2U[b]) for b in text.encode("utf-8"))


def bpe(token, ranks):
    word = list(token)
    while len(word) > 1:
        pairs = [(ranks.get(word[i] + " " + word[i + 1]), i) for i in range(len(word) - 1)]
        pairs = [p for p in pairs if p[0] is not None]
        if not pairs:
            break
        best = min(pairs)[0]
        i = 0
        while i < len(word) - 1:
            if ranks.get(word[i] + " " + word[i + 1]) == best:
                word[i:i + 2] = [word[i] + word[i + 1]]
            else:
                i += 1
    return word


def encode(text):
    ranks = {m: i for i, m in enumerate(MERGES)}
    ids = []
    for piece in PAT.findall(text):
        for symbol in bpe(byte_encode(piece), ranks):
            ids.append(VOCAB[symbol])
    return ids


def decode(ids):
    out = b""
    for i in ids:
        for ch in build_vocab()[i]:
            b = B2U_REV.get(ord(ch))
            out += bytes([b]) if b is not None else ch.encode("utf-8")
    return out


B2U_REV = {cp: b for b, cp in B2U.items()}

CASES = [
    "hello world",
    "hello",
    "hello  world",
    "  hi",
    "Hello",
    "don't",
    "a1!?",
    "it's we're I'll",
    "\n\nHello",
    "hello\nworld",
    "  ",
    "x",
    "日本語",
    "café",
    "naïve café",
    "hello, world!",
    "trailing ",
    " leading",
]

if __name__ == "__main__":
    print("pre-tokens:")
    for c in CASES:
        print("  %-18r -> %r" % (c, PAT.findall(c)))
    print("encode:")
    for c in CASES:
        ids = encode(c)
        assert decode(ids).decode("utf-8") == c, (c, ids, decode(ids))
        print("  %-18r -> %s" % (c, ids))
