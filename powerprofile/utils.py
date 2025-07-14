# -*- coding: utf-8 -*-
import math
from decimal import Decimal
from datetime import datetime
try:
    from collections import Counter
except ImportError:
    from backport_collections import Counter


class Dragger(Counter):

    def drag(self, number, key='default'):
        if number == 0 and abs(self[key]) == Decimal('0.5'):
            # Avoid oscillation between -1 and 1 and dragging 0.5 and -0.5
            return number

        number = Decimal(str(number)) + self[key]
        aprox = int(round(number))
        self[key] = number - aprox
        return aprox


def interpolate_quarter_curve(values):
    """
    Interpola valors horaris en valors quarthoraris seguint el procediment
    escrit a l'Annex 11 del BOE.

    Parameters:
    values : list of int or float
        Llista ordenada de valors horaris on:
        - values[1] fins a values[-2] representen els valors d'energia horària (E_h)
        - values[0] és el valor inicial auxiliar (E_{h-1}) per la primera hora
        - values[-1] és el valor final auxiliar (E_{h+1}) per l’última hora

    Returns:
    Generator
        Una llista amb un diccionari per cada quart d’hora interpolat amb informació detallada.
    """

    def get_eh_idx(hour, quarter):
        """
        Determina quins valors horaris s’utilitzen per interpolar segons si el quart és el 1r/2n o 3r/4t.
        """
        return (hour, hour - 1) if quarter in (1, 2) else (hour + 1, hour)

    def get_xqh(hour, quarter):
        """
        Calcula la posició temporal (abscissa) del quart d’hora dins del bloc horari.
        """
        return (hour - 1) + 0.125 + 0.25 * (quarter - 1)

    def get_xhd(hour, quarter):
        """
        Retorna les posicions abscisses x_{h+1} i x_{h-1} segons la posició del quart.
        """
        hour -= 1
        if quarter in (1, 2):
            return (hour + 0.5, hour - 0.5)
        else:
            return (hour + 1.5, hour + 0.5)

    # Si els extrems no estan inicialitzats, se substitueixen pel veí immediat
    if values[0] is None:
        values[0] = values[1]
    if values[-1] is None:
        values[-1] = values[-2]

    result = []
    qt = 1  # Índex global de quart-hora

    # Iterem per cada hora central (les reals a interpolar)
    for h in range(1, len(values) - 1):
        sum_ehd = 0.0
        qh_data = {}

        # Interpolació lineal segons fórmula oficial: E'_{qhd}
        for q in range(1, 5):
            hi, hm = get_eh_idx(h, q)
            xqh = get_xqh(h, q)
            xhd, xhd_1 = get_xhd(h, q)

            ehd = values[hm] + ((values[hi] - values[hm]) / (xhd - xhd_1)) * (xqh - xhd_1)
            eqhd = ehd / 4.0

            qh_data[q] = {
                'hour': h,
                'quarter': q,
                'qt_index': qt,
                'xqh': xqh,
                'ehd': ehd,
                'eqhd': eqhd,
            }

            sum_ehd += eqhd
            qt += 1

        # Ajust per assegurar que la suma de quarts sigui exactament igual a E_h
        diff = values[h] - sum_ehd
        rounded_sum = 0

        for q in range(1, 5):
            # norm_qh = qh_data[q]['eqhd'] + ((diff * qh_data[q]['eqhd']) / sum_ehd)
            if sum_ehd:
                norm_qh = qh_data[q]['eqhd'] + ((diff * qh_data[q]['eqhd']) / sum_ehd)
            else:
                norm_qh = 0
            if q < 4:
                round_qh = int(round(norm_qh))
                rounded_sum += round_qh
            else:
                round_qh = int(values[h] - rounded_sum)  # Últim quart compensa l’ajust

            qh_data[q].update({
                'diff': diff,
                'norm_qh': norm_qh,
                'round_qh': round_qh,
            })
            yield qh_data[q]

def my_round(x, d=0):
    x = float(x)
    p = 10 ** d
    if x > 0:
        return float(math.floor((x * p) + 0.5))/p
    else:
        return float(math.ceil((x * p) - 0.5))/p
