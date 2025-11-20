import sys

for line in sys.stdin:
    s = line.strip()
    if not s:
        continue
    n = int(s, 16)
    sign = (n >> 31) & 1
    exp = (n >> 23) & 0xff
    mant = n & 0x7fffff

    if exp == 0xff and mant != 0:
        break

    if exp == 0 and mant == 0:
        print(f"{'-' if sign else ''}0.0")
        continue

    if exp == 0:
        k = mant.bit_length() - 1
        frac_bits = ''.join('1' if (mant >> i) & 1 else '0' for i in range(k - 1, -1, -1))
        frac_bits = frac_bits.rstrip('0')
        frac_bits = frac_bits if frac_bits else '0'
        e = k - 149
        print(f"{'-' if sign else ''}1.{frac_bits} x 2 ** {e}")
        continue

    e = exp - 127
    bits = ''.join('1' if mant & (1 << (22 - i)) else '0' for i in range(23))
    frac_bits = bits.rstrip('0')
    frac_bits = frac_bits if frac_bits else '0'
    print(f"{'-' if sign else ''}1.{frac_bits} x 2 ** {e}")
