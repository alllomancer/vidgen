"""cli/qr.py, the dashboard's QR encoder (the "Pair a phone" link). Cross-checked once against segno 1.6.1: all of
versions 1-10 x 8 masks match module for module, except that segno adds one extra 0x00 pad byte after the
terminator where ISO 18004 7.4.10 pads only to the next codeword boundary (both decode the same). This test pins
those verified symbols and checks the structure a scanner relies on."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli"))
import qr  # noqa: E402

URL = "https://abcdefghij.execute-api.us-east-1.amazonaws.com/m"
PINNED = {URL: (33, "ffa4d4ab9fcc4bb1"), "h": (21, "848f621f7ecfca7e"), "x" * 120: (45, "b91b7e904d3e58a9")}
for text, (size, digest) in PINNED.items():
    rows = qr.encode(text)
    assert len(rows) == size and all(len(r) == size for r in rows), (text, len(rows))
    assert hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16] == digest, text


def finder_ok(rows, x0, y0):
    want = ["1111111", "1000001", "1011101", "1011101", "1011101", "1000001", "1111111"]
    return [rows[y0 + i][x0:x0 + 7] for i in range(7)] == want


def format_info(rows):
    """(level bits, mask) from the copy around the top-left finder, after a BCH check."""
    bit = lambda x, y: int(rows[y][x])  # noqa: E731
    bits = [bit(8, i) for i in range(6)] + [bit(8, 7), bit(8, 8), bit(7, 8)] + [bit(14 - i, 8) for i in range(9, 15)]
    value = sum(b << i for i, b in enumerate(bits)) ^ 0x5412
    data, rem = value >> 10, value
    for i in range(14, 9, -1):
        if rem >> i & 1:
            rem ^= 0x537 << (i - 10)
    assert rem == 0, "format bits fail their BCH check"
    return data >> 3, data & 7


for text in (URL, "h", "x" * 120, "é" * 20):
    rows = qr.encode(text)
    n = len(rows)
    assert finder_ok(rows, 0, 0) and finder_ok(rows, n - 7, 0) and finder_ok(rows, 0, n - 7), text
    assert rows[6][8:n - 8] == "".join("10"[i % 2] for i in range(8, n - 8)), "timing row"
    assert rows[n - 8][8] == "1", "the dark module"
    level, mask = format_info(rows)
    assert level == 0, "level M"
    for k in range(8):  # every mask gives a valid symbol that says which mask it is
        assert format_info(qr.encode(text, mask=k)) == (0, k)

# the smallest version that fits is used; too long is refused
assert len(qr.encode("a" * 14)) == 21 and len(qr.encode("a" * 15)) == 25
try:
    qr.encode("a" * 214)
    raise AssertionError("v10-M holds 213 bytes at most")
except ValueError:
    pass

print("test_qr: ok")
