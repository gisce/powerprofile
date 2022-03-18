# -*- coding: utf-8 -*-
from expects import *
from powerprofile.exceptions import *


except_data = [
    (PowerProfileDuplicatedTimes, "PowerProfile Duplicated Times"),
    (PowerProfileIncompatible, "PowerProfile Incompatible"),
    (PowerProfileIncompleteCurve, "PowerProfile Incomplete Curve"),
    (PowerProfileNotImplemented, "Operation not implemented"),
    (PowerProfileMissingField, "Field does not exist in profile: "),
    (PowerProfileNegativeCurve, "PowerProfile Negative Curve")
]


with description('PowerProfile Exceptions'):

    with context('Raise it and'):

        with it('returns the text'):

            for exc, txt in except_data:
                try:
                    raise exc('This is the text')
                except Exception as e:
                    text = str(e)

                expect(text).to(contain('This is the text'))
                expect(text).to(contain(txt))