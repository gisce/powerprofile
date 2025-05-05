# -*- coding: utf-8 -*-
from expects.testing import failure
from mamba import *
from expects import *
from powerprofile.powerprofile import PowerProfile, PowerProfileQh, DEFAULT_DATA_FIELDS
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_datetime
from pytz import timezone
from copy import copy, deepcopy
from powerprofile.exceptions import *
import json
import random
try:
    # Python 2
    from cStringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO
import csv
import pandas as pd

LOCAL_TZ = timezone('Europe/Madrid')
UTC_TZ = timezone('UTC')

with description('PowerProfileQh class'):

    with context('Instance'):

        with it('Returns PowerProfile Object'):
            powerprofile = PowerProfileQh()

            expect(powerprofile).to(be_a(PowerProfileQh))
    with context('load function'):
        with context('with bad data'):
            with it('raises TypeError exception'):
                powerprofile = PowerProfileQh()

                expect(lambda: powerprofile.load({'timestamp': '2020-01-31 10:00:00',  "ai": 13.0})).to(
                    raise_error(TypeError, "ERROR: [data] must be a list of dicts ordered by timestamp")
                )
                expect(lambda: powerprofile.load(
                    [{'timestamp': '2020-01-31 10:00:00',  "ai": 13.0}], start='2020-03-11')).to(
                    raise_error(TypeError, "ERROR: [start] must be a localized datetime")
                )
                expect(lambda: powerprofile.load(
                    [{'timestamp': '2020-01-31 10:00:00',  "ai": 13.0}], end='2020-03-11')).to(
                    raise_error(TypeError, "ERROR: [end] must be a localized datetime")
                )

        with context('correctly'):
            with before.all:
                self.curve = []
                self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 0, 15, 0))
                self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
                for hours in range(0, 24):
                    for minutes in [0, 15, 30, 45]:
                        self.curve.append({'timestamp': self.start + timedelta(minutes=hours*60 + minutes), 'value': 100 + hours * 10 + minutes})
                self.powpro = PowerProfileQh()

            with it('without dates'):
                self.powpro.load(self.curve)
                expect(lambda: self.powpro.check()).not_to(raise_error)
                expect(self.powpro.data_fields).to(equal(['value']))

            with it('with start date'):
                self.powpro.load(self.curve, start=self.start)
                expect(lambda: self.powpro.check()).not_to(raise_error)
                expect(self.powpro.data_fields).to(equal(['value']))

            with it('with end date'):
                self.powpro.load(self.curve, end=self.end)
                expect(lambda: self.powpro.check()).not_to(raise_error)
                expect(self.powpro.data_fields).to(equal(['value']))

            with it('with start and end date'):
                self.powpro.load(self.curve, start=self.start, end=self.end)
                expect(lambda: self.powpro.check()).not_to(raise_error)
                expect(self.powpro.data_fields).to(equal(['value']))

            with it('with datetime_field field in load'):

                curve_name = []
                for row in self.curve:
                    new_row = copy(row)
                    new_row['datetime'] = new_row['timestamp']
                    new_row.pop('timestamp')
                    curve_name.append(new_row)

                powpro = PowerProfileQh()

                expect(lambda: powpro.load(curve_name)).to(raise_error(TypeError))

                expect(lambda: powpro.load(curve_name, datetime_field='datetime')).to_not(raise_error(TypeError))
                expect(powpro[0]).to(have_key('datetime'))

            with it('with datetime_field field in constructor'):

                curve_name = []
                for row in self.curve:
                    new_row = copy(row)
                    new_row['datetime'] = new_row['timestamp']
                    new_row.pop('timestamp')
                    curve_name.append(new_row)

                powpro = PowerProfileQh(datetime_field='datetime')

                expect(lambda: powpro.load(curve_name)).to_not(raise_error(TypeError))
                expect(powpro[0]).to(have_key('datetime'))

            with it('with data_fields field in load'):

                powpro = PowerProfileQh()

                powpro.load(self.curve, data_fields=['value'])
                expect(powpro.data_fields).to(equal(['value']))

                expect(lambda: powpro.load(curve_name, data_fields=['value'])).to_not(raise_error(TypeError))
                expect(powpro[0]).to(have_key('value'))

        with context('with unlocalized datetimes'):
            with before.all:
                self.curve = []
                self.start = datetime(2022, 8, 1, 1, 0, 0)
                self.end = datetime(2022, 9, 1, 0, 0, 0)
                for hours in range(0, 24):
                    self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})
            with it('should localize datetimes on load'):
                powerprofile = PowerProfileQh()
                powerprofile.load(self.curve)
                for idx, hour in powerprofile.curve.iterrows():
                    assert hour[powerprofile.datetime_field].tzinfo is not None

    with context('dump function'):
        with before.all:
            self.curve = []
            self.erp_curve = []
            self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
            self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
            for hours in range(0, 24):
                for minutes in [15, 30, 45, 60]:
                    self.curve.append({'timestamp': self.start + timedelta(hours=hours, minutes=minutes), 'value': 100 + hours*10 + minutes})
                    self.erp_curve.append(
                        {
                            'local_datetime': self.start + timedelta(hours=hours, minutes=minutes),
                            'utc_datetime': (self.start + timedelta(hours=hours, minutes=minutes)).astimezone(UTC_TZ),
                            'value': 100 + hours,
                            'valid': bool(hours % 2),
                            'period': 'P' + str(hours % 3),
                        }
                    )

        with context('performs complet curve -> load -> dump -> curve circuit '):

            with it('works with simple format'):
                powpro = PowerProfileQh()
                powpro.load(self.curve)
                curve = powpro.dump()
                expect(curve).to(equal(self.curve))

            with it('works with ERP curve API'):
                powpro = PowerProfileQh(datetime_field='utc_datetime')
                powpro.load(self.erp_curve)
                erp_curve = powpro.dump()

                expect(erp_curve).to(equal(self.erp_curve))

    with description('PowerProfileQH Operators'):
        with before.all:
            self.curve = []
            self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 0, 15, 0))
            self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
            for hours in range(0, 24):
                for minutes in [0, 15, 30, 45]:
                    self.curve.append({'timestamp': self.start + timedelta(minutes=hours * 60 + minutes),
                                       'value': 100 + hours * 10 + minutes})
            self.powpro = PowerProfileQh()
            self.powpro.load(self.curve)

        with description('Unary Operator'):
            with context('get_hourly_profile'):
                with it('returns a PowerProfile instance'):
                    pph = self.powpro.get_hourly_profile()

                    expect(pph).to(be_an(PowerProfile))
                    expect(pph).not_to(be_an(PowerProfileQh))

                    pph.check()
                    expect(pph.start).to(equal(LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))))
                    expect(pph.end).to(equal(self.powpro.end))
                    expect(pph.sum(['value'])).to(equal(self.powpro.sum(['value'])))
                    expect(pph.hours).to(equal(24))
                    expected_value = sum([x['value'] for x in self.powpro[:4]])
                    expect(pph[0]['value']).to(equal(expected_value))

    with description('transform to quarter-hour curve'):
        with before.all as self:
            self.curve = []
            self.start = LOCAL_TZ.localize(datetime(2023, 3, 11, 1, 0, 0))
            for h in range(24):
                self.curve.append({'timestamp': self.start + timedelta(hours=h),
                                   'value': 100 + h})
            self.hourly = PowerProfile()
            self.hourly.load(self.curve)

        with it('returns PowerProfileQh with 96 samples'):
            qh = self.hourly.to_qh()
            expect(qh).to(be_a(PowerProfileQh))
            expect(qh.quart_hours).to(equal(24 * 4))  # 24 hores × 4 quarts

        with it('ensures interpolation is consistent by hour'):
            qh = self.hourly.to_qh()
            qh_dict = qh.curve.set_index('timestamp').to_dict()['value']
            for row in self.curve:
                ts = row['timestamp']
                expected = row['value']
                # El consum de la hora 3 representa el consum de la hora 2 fins a la hora 3 llavors, per la hora 3
                # al interpolar hauriem d'obtenir els quarts d'hora 2:15, 2:30, 2:45, 3:00. Per aixo a sota restem una
                # hora a la hora de la corba horaria.
                qts = [
                    ts - timedelta(hours=1) + timedelta(minutes=i * 15) for i in range(1, 5)
                ]
                interpolated_sum = sum(qh_dict[qt] for qt in qts)
                expect(interpolated_sum).to(equal(expected))
