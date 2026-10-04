"""Sparse rational Taylor coefficients, retaining real scalar arithmetic."""
from mpmath import mp


def polynomial_value(coefficients, point):
    value = mp.zero
    for coefficient in reversed(coefficients):
        value = coefficient + point * value
    return value


def _shift(coefficients, point):
    """Coefficients of p(point + u), in ascending powers of u."""
    shifted = [coefficients[-1]]
    for coefficient in reversed(coefficients[:-1]):
        shifted.append(mp.zero)
        for k in range(len(shifted) - 1, 0, -1):
            shifted[k] = shifted[k - 1] + point * shifted[k]
        shifted[0] = coefficient + point * shifted[0]
    return shifted


def rational_series(entries, point, order):
    """Expand each distinct rational function once for this point and order."""
    if not mp.im(point):
        point = mp.re(point)
    inverses, series, answer = {}, {}, {}
    for key, (numerator, denominator, scale) in entries.items():
        # Normalization varies by matrix entry; cache only unscaled terms.
        # This cache is local, so another point, order, or precision rebuilds it.
        function_key = numerator, denominator
        coefficients = series.get(function_key)
        if coefficients is None:
            inverse = inverses.get(denominator)
            if inverse is None:
                shifted = _shift(denominator, point)
                if shifted[0] == 0:
                    raise ZeroDivisionError("Rational connection has a pole at the expansion point")
                inverse_constant = 1 / shifted[0]
                inverse = [inverse_constant]
                for k in range(1, order + 1):
                    value = sum((shifted[j] * inverse[k - j]
                                 for j in range(1, min(k, len(shifted) - 1) + 1)), mp.zero)
                    inverse.append(-value * inverse_constant)
                inverses[denominator] = inverse
            shifted = _shift(numerator, point)
            coefficients = [sum((shifted[j] * inverse[k - j]
                                 for j in range(min(k, len(shifted) - 1) + 1)), mp.zero)
                            for k in range(order + 1)]
            series[function_key] = coefficients
        answer[key] = [scale * coefficient for coefficient in coefficients]
    return answer
