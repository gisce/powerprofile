# coding: utf-8
from mamba import *
from expects import *
from powerprofile.utils import interpolate_quarter_curve


with description('interpolate_quarter_curve'):
    with context('REE simulator validation'):
        with it('generates exact quarter-hour results from REE simulator'):

            values = [
                1105, 1279, 1354, 1338, 1327, 1386, 1471, 1590, 1639, 1791, 1788, 1820,
                1776, 1742, 1795, 1653, 1511, 1441, 1596, 1604, 1508, 1429, 1477, 1390,
                1326, 1258,
            ]

            # Valors de referència extrets del simulador (ajust final)
            expected_round_qh = [
                306, 317, 325, 331, 334, 339, 341, 340, 336, 335, 334, 333, 331,
                330, 331, 335, 340, 344, 348, 354, 359, 364, 370, 378, 388, 396,
                401, 405, 402, 405, 411, 421,438, 448, 453, 452, 446, 446, 447,
                449, 454, 456, 456, 454, 448, 445, 443, 440, 436, 434, 434, 438,
                450, 453, 450, 442, 427, 418, 409, 399, 389, 380, 373, 369, 360,
                355, 358, 368, 389, 399, 404, 404, 404, 404, 401, 395, 385, 379,
                374, 370, 361, 356, 355, 357, 369, 372, 371, 365, 355, 349, 345,
                341, 338, 334, 329, 325,
            ]

            # També podries incloure els valors "norm_qh" si vols validar precisió

            result = list(interpolate_quarter_curve(values))
            actual_round_qh = [item['round_qh'] for item in result]

            expect(actual_round_qh).to(equal(expected_round_qh))
        with it('generates exact quarter-hour results from REE simulator with 3 hours in a row to 0'):

            values = [
                1105, 0, 0, 0, 1327, 1386, 1471, 1590, 1639, 1791, 1788, 1820,
                1776, 1742, 1795, 1653, 1511, 1441, 1596, 1604, 1508, 1429, 1477, 1390,
                1326, 1258,
            ]
            # Valors de referència extrets del simulador (ajust final)
            expected_round_qh = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 235,
                330, 379, 383, 340, 344, 348, 354, 359, 364, 370, 378, 388, 396,
                401, 405, 402, 405, 411, 421, 438, 448, 453, 452, 446, 446, 447,
                449, 454, 456, 456, 454, 448, 445, 443, 440, 436, 434, 434, 438,
                450, 453, 450, 442, 427, 418, 409, 399, 389, 380, 373, 369, 360,
                355, 358, 368, 389, 399, 404, 404, 404, 404, 401, 395, 385, 379,
                374, 370, 361, 356, 355, 357, 369, 372, 371, 365, 355, 349, 345,
                341, 338, 334, 329, 325,
            ]

            # També podries incloure els valors "norm_qh" si vols validar precisió
            result = list(interpolate_quarter_curve(values))
            actual_round_qh = [item['round_qh'] for item in result]

            expect(actual_round_qh).to(equal(expected_round_qh))