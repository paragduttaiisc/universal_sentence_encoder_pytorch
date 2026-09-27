"""Python port of TF's StringToHashBucketFast primitive: farmhash Fingerprint64.

TF 2.x StringToHashBucketFast computes  Fingerprint64(utf8_bytes) % num_buckets
where Fingerprint64 is Google FarmHash's stable fingerprint
(google/farmhash, namespace util / farmhashna, Hash64, little-endian fetch).
Verified bit-exact against the C++ farmhash build and against TF 2.15's
tf.raw_ops.StringToHashBucketFast (see porting/fp_check.py).
All arithmetic is mod 2^64.
"""

_M64 = (1 << 64) - 1
_K0 = 0xC3A5C85C97CB3127
_K1 = 0xB492B66FBE98F273
_K2 = 0x9AE16A3B2F90404F
_KMUL = 0x9DDFEA08EB382D69


def _rot64(x: int, s: int) -> int:
    """FarmHash Rotate64: right-rotate mod 2^64."""
    s %= 64
    if s == 0:
        return x
    return ((x >> s) | (x << (64 - s))) & _M64


def _shift_mix(x: int) -> int:
    return x ^ (x >> 47)


def _fetch64(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i + 8], "little")


def _fetch32(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i + 4], "little")


def _hash128to64(u: int, v: int) -> int:
    a = ((u ^ v) * _KMUL) & _M64
    a ^= a >> 47
    b = ((v ^ a) * _KMUL) & _M64
    b ^= b >> 47
    return (b * _KMUL) & _M64


def _hashlen16(u: int, v: int, mul: int) -> int:
    a = ((u ^ v) * mul) & _M64
    a ^= a >> 47
    b = ((v ^ a) * mul) & _M64
    b ^= b >> 47
    return (b * mul) & _M64


def _weak32(w: int, x: int, y: int, z: int, a: int, b: int) -> tuple:
    a = (a + w) & _M64
    b = _rot64((b + a + z) & _M64, 21)
    c = a
    a = (a + x) & _M64
    a = (a + y) & _M64
    b = (b + _rot64(a, 44)) & _M64
    return ((a + z) & _M64, (b + c) & _M64)


def _weak32_str(s: bytes, off: int, a: int, b: int) -> tuple:
    return _weak32(_fetch64(s, off), _fetch64(s, off + 8),
                   _fetch64(s, off + 16), _fetch64(s, off + 24), a, b)


def _hashlen0to16(s: bytes, n: int) -> int:
    if n >= 8:
        mul = _K2 + n * 2
        a = (_fetch64(s, 0) + _K2) & _M64
        b = _fetch64(s, n - 8)
        c = (_rot64(b, 37) * mul + a) & _M64
        d = ((_rot64(a, 25) + b) & _M64) * mul & _M64
        return _hashlen16(c, d, mul)
    if n >= 4:
        mul = _K2 + n * 2
        a = _fetch32(s, 0)
        return _hashlen16((n + (a << 3)) & _M64, _fetch32(s, n - 4), mul)
    if n > 0:
        a = s[0]
        b = s[n >> 1]
        c = s[n - 1]
        y = a + (b << 8)
        z = n + (c << 2)
        t = ((y * _K2) & _M64) ^ ((z * _K0) & _M64)
        return (_shift_mix(t) * _K2) & _M64
    return _K2


def _hashlen17to32(s: bytes, n: int) -> int:
    mul = _K2 + n * 2
    a = (_fetch64(s, 0) * _K1) & _M64
    b = _fetch64(s, 8)
    c = (_fetch64(s, n - 8) * mul) & _M64
    d = (_fetch64(s, n - 16) * _K2) & _M64
    u = (_rot64((a + b) & _M64, 43) + _rot64(c, 30) + d) & _M64
    v = (a + _rot64((b + _K2) & _M64, 18) + c) & _M64
    return _hashlen16(u, v, mul)


def _hashlen33to64(s: bytes, n: int) -> int:
    mul = _K2 + n * 2
    a = (_fetch64(s, 0) * _K2) & _M64
    b = _fetch64(s, 8)
    c = (_fetch64(s, n - 8) * mul) & _M64
    d = (_fetch64(s, n - 16) * _K2) & _M64
    y = (_rot64((a + b) & _M64, 43) + _rot64(c, 30) + d) & _M64
    z = _hashlen16(y, (a + _rot64((b + _K2) & _M64, 18) + c) & _M64, mul)
    e = (_fetch64(s, 16) * mul) & _M64
    f = _fetch64(s, 24)
    g = ((y + _fetch64(s, n - 32)) & _M64) * mul & _M64
    h = ((z + _fetch64(s, n - 24)) & _M64) * mul & _M64
    u = (_rot64((e + f) & _M64, 43) + _rot64(g, 30) + h) & _M64
    v = (e + _rot64((f + a) & _M64, 18) + g) & _M64
    return _hashlen16(u, v, mul)


def _hash64_long(s: bytes, n: int) -> int:
    seed = 81
    x = seed
    y = (seed * _K1 + 113) & _M64
    z = (_shift_mix((y * _K2 + 113) & _M64) * _K2) & _M64
    v0, v1 = 0, 0
    w0, w1 = 0, 0
    x = (x * _K2 + _fetch64(s, 0)) & _M64

    off = 0
    end = ((n - 1) // 64) * 64
    last64 = end + ((n - 1) & 63) - 63
    while off != end:
        x = (_rot64((x + y + v0 + _fetch64(s, off + 8)) & _M64, 37) * _K1) & _M64
        y = (_rot64((y + v1 + _fetch64(s, off + 48)) & _M64, 42) * _K1) & _M64
        x ^= w1
        y = (y + v0 + _fetch64(s, off + 40)) & _M64
        z = (_rot64((z + w0) & _M64, 33) * _K1) & _M64
        v0, v1 = _weak32_str(s, off, (v1 * _K1) & _M64, (x + w0) & _M64)
        w0, w1 = _weak32_str(s, off + 32, (z + w1) & _M64, (y + _fetch64(s, off + 16)) & _M64)
        z, x = x, z
        off += 64

    mul = (_K1 + (((z & 0xFF) << 1))) & _M64
    off = last64
    w0 = (w0 + ((n - 1) & 63)) & _M64
    v0 = (v0 + w0) & _M64
    w0 = (w0 + v0) & _M64
    x = (_rot64((x + y + v0 + _fetch64(s, off + 8)) & _M64, 37) * mul) & _M64
    y = (_rot64((y + v1 + _fetch64(s, off + 48)) & _M64, 42) * mul) & _M64
    x ^= (w1 * 9) & _M64
    y = (y + v0 * 9 + _fetch64(s, off + 40)) & _M64
    z = (_rot64((z + w0) & _M64, 33) * mul) & _M64
    v0, v1 = _weak32_str(s, off, (v1 * mul) & _M64, (x + w0) & _M64)
    w0, w1 = _weak32_str(s, off + 32, (z + w1) & _M64, (y + _fetch64(s, off + 16)) & _M64)
    z, x = x, z
    a = (_hashlen16(v0, w0, mul) + _shift_mix(y) * _K0 + z) & _M64
    b = (_hashlen16(v1, w1, mul) + x) & _M64
    return _hashlen16(a, b, mul)


def _fp64_bytes(data: bytes) -> int:
    n = len(data)
    if n <= 16:
        return _hashlen0to16(data, n)
    if n <= 32:
        return _hashlen17to32(data, n)
    if n <= 64:
        return _hashlen33to64(data, n)
    return _hash64_long(data, n)


def fingerprint64(s: str) -> int:
    """TF Fingerprint64 / StringToHashBucketFast base hash over UTF-8 bytes."""
    return _fp64_bytes(s.encode("utf-8"))


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        print(f"{w!r:40} {fingerprint64(w):016x}  mod200k={fingerprint64(w) % 200000}")
