# -*- coding: utf-8 -*-
from expects.testing import failure
from expects import *
from powerprofile.powerprofile import PowerProfile, DEFAULT_DATA_FIELDS
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_datetime
from pytz import timezone
from copy import copy
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


def datetime_parser(dct):
    for k, v in dct.items():
        # local datetime fields
        if k in ['timestamp', 'local_datetime', 'utc_datetime']:
            try:
                dct[k] = parse_datetime(v)
            except Exception as e:
                pass
    return dct


def create_test_curve(start_date, end_date):
    """
    From two local_time_dates, create a hourly_curve like ERP api
    :param start_date: local datetime
    :param end_date: local datetime
    :return: list of dicts as in ERP get_hourly_curve API
    """
    curve = []
    cur_date = start_date
    while cur_date <= end_date:
        value = random.uniform(1.0, 200.0)
        curve.append({'utc_datetime': cur_date, 'value': value})
        cur_date += relativedelta(hours=1)

    return curve


def read_csv(txt):
    """
    returns an list of list with csv ';' delimited content
    :param txt: the csv ';' delimited format string
    :return: a list or rows as list
    """
    f = StringIO(txt)
    reader = csv.reader(f, delimiter=';')
    csv_curve = []
    for row in reader:
        csv_curve.append(row)

    return csv_curve


with description('PowerProfile class'):

    with context('Instance'):

        with it('Returns PowerProfile Object'):
            powerprofile = PowerProfile()

            expect(powerprofile).to(be_a(PowerProfile))
    with context('load function'):
        with context('with bad data'):
            with it('raises TypeError exception'):
                powerprofile = PowerProfile()

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
                self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
                self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
                for hours in range(0, 24):
                    self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})
                self.powpro = PowerProfile()

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

                powpro = PowerProfile()

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

                powpro = PowerProfile(datetime_field='datetime')

                expect(lambda: powpro.load(curve_name)).to_not(raise_error(TypeError))
                expect(powpro[0]).to(have_key('datetime'))

            with it('with data_fields field in load'):

                powpro = PowerProfile()

                powpro.load(self.curve, data_fields=['value'])
                expect(powpro.data_fields).to(equal(['value']))

                expect(lambda: powpro.load(curve_name, data_fields=['value'])).to_not(raise_error(TypeError))
                expect(powpro[0]).to(have_key('value'))

    with context('dump function'):
        with before.all:
            self.curve = []
            self.erp_curve = []
            self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
            self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
            for hours in range(0, 24):
                self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})
                self.erp_curve.append(
                    {
                        'local_datetime': self.start + timedelta(hours=hours),
                        'utc_datetime': (self.start + timedelta(hours=hours)).astimezone(UTC_TZ),
                        'value': 100 + hours,
                        'valid': bool(hours % 2),
                        'period': 'P' + str(hours % 3),
                    }
                )

        with context('performs complet curve -> load -> dump -> curve circuit '):

            with it('works with simple format'):
                powpro = PowerProfile()
                powpro.load(self.curve)
                curve = powpro.dump()
                expect(curve).to(equal(self.curve))

            with it('works with ERP curve API'):
                powpro = PowerProfile(datetime_field='utc_datetime')
                powpro.load(self.erp_curve)
                erp_curve = powpro.dump()

                expect(erp_curve).to(equal(self.erp_curve))


    with context('check curve'):
        with before.all:
            self.curve = []
            self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
            self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
            for hours in range(0, 24):
                self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})

            self.original_curve_len = len(self.curve)
            self.powpro = PowerProfile()

        with context('completeness'):

            with it('returns true when complete'):
                self.powpro.load(self.curve)

                expect(self.powpro.is_complete()).to(be_true)
                expect(lambda: self.powpro.check()).not_to(raise_error)

            with it('returns false when hole'):
                curve = copy(self.curve)
                del curve[3]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len - 1))
                expect(self.powpro.is_complete()).to(be_false)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileIncompleteCurve))

            with it('returns false when hole at beginning'):
                curve = copy(self.curve)
                del curve[0]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len - 1))
                expect(self.powpro.is_complete()).to(be_false)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileIncompleteCurve))

            with it('returns false when hole at end'):
                curve = copy(self.curve)
                del curve[0]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len - 1))
                expect(self.powpro.is_complete()).to(be_false)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileIncompleteCurve))

        with context('duplicated hours'):

            with it('returns true when not duplicates'):
                self.powpro.load(self.curve)

                expect(self.powpro.has_duplicates()).to(be_false)
                expect(lambda: self.powpro.check()).not_to(raise_error)

            with it('returns false when duplicates extra hour'):
                curve = copy(self.curve)
                curve.append(self.curve[3])
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len + 1))
                expect(self.powpro.has_duplicates()).to(be_true)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileDuplicatedTimes))

            with it('returns false when duplicates and correct length'):
                curve = copy(self.curve)
                curve.append(self.curve[3])
                del curve[0]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len))
                expect(self.powpro.has_duplicates()).to(be_true)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileDuplicatedTimes))

    with context('accessing data'):
        with before.all:
            self.curve = []
            self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
            self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
            for hours in range(0, 24):
                self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})

            self.powpro = PowerProfile()
            self.powpro.load(self.curve)

        with context('by index operator [int:int]'):
            with context('correctly'):
                with it('returns a dict when index'):
                    for v in range(len(self.curve) - 1):
                        res = self.powpro[v]

                        expect(res).to(equal(self.curve[v]))

                with it('returns a new small powerprofile on int slice'):
                    res = self.powpro[1:4]

                    expect(res).to(be_a(PowerProfile))
                    expect(res.hours).to(equal(3))
                    expect(res.start).to(equal(self.curve[1]['timestamp']))
                    expect(res.end).to(equal(self.curve[3]['timestamp']))
                    for v in 1, 2, 3:
                        res[v-1] == self.powpro[v]

            with context('when bad index'):
                with it('raises IndexError (key)'):
                    expect(lambda: self.powpro[len(self.curve) + 1]).to(raise_error(IndexError))

                with it('raises IndexError (slice)'):
                    expect(lambda: self.powpro[1:50]).to(raise_error(IndexError))

        with context('by timestamp [datetime]'):

            with context('correctly'):

                with it('returns a dict when localized datetime'):
                    dt = LOCAL_TZ.localize(datetime(2020, 3, 11, 2, 0, 0))
                    res = self.powpro[dt]

                    expect(res).to(equal(self.curve[1]))

            with context('when bad datetime'):
                with it('raises TypeError when naive datetime'):
                    dt = datetime(2020, 3, 11, 2, 0, 0)
                    expect(lambda: self.powpro[dt]).to(raise_error(TypeError))

    with context('Aggregation operators'):
        with before.all:
            self.curve = []
            self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
            self.end = LOCAL_TZ.localize(datetime(2020, 3, 12, 0, 0, 0))
            for hours in range(0, 24):
                self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})

            self.powpro = PowerProfile()
            self.powpro.load(self.curve)

        with context('total sum'):

            with it('returns sum of magnitudes in curve'):

                res = self.powpro.sum(['value'])

                total_curve = sum([v['value'] for v in self.curve])

                expect(res['value']).to(equal(total_curve))

    with context('Real curves'):
        with before.all:
            self.data_path = './spec/data/'

        with context('curve.json'):

            with it('return correct powerprofile object'):

                with open(self.data_path + 'erp_curve.json') as fp:
                    erp_curve = json.load(fp, object_hook=datetime_parser)

                curve = erp_curve['curve']
                datetime_field = 'utc_datetime'
                powpro = PowerProfile(datetime_field)
                powpro.load(curve, curve[0][datetime_field], curve[-1][datetime_field])
                totals = powpro.sum(['ae', 'ai'])

                expect(powpro.check()).to(be_true)
                expect(totals['ai']).to(be_above(0))
                expect(totals['ae']).to(be_above(0))

                dumped_curve = powpro.dump()
                expect(dumped_curve).to(equal(curve))

with description('PowerProfile Manipulation'):
    with before.all:
        self.data_path = './spec/data/'

        with open(self.data_path + 'erp_curve.json') as fp:
            self.erp_curve = json.load(fp, object_hook=datetime_parser)

    with fcontext('Self transformation functions'):
        with context('Balance'):
            with it('Performs a by hourly Balance between two magnitudes and stores in ac postfix columns'):
                powpro = PowerProfile()
                powpro.load(self.erp_curve['curve'], datetime_field='utc_datetime')

                powpro.Balance('ae', 'ai')

                expect(powpro.check()).to(be_true)
                for i in range(powpro.hours):
                    row = powpro[i]
                    if row['ae'] >= row['ai']:
                        expect(row['aebal']).to(equal(row['ae'] - row['ai']))
                        expect(row['aibal']).to(equal(0.0))
                    else:
                        expect(row['aibal']).to(equal(row['ai'] - row['ae']))
                        expect(row['aebal']).to(equal(0.0))

        with context('Min'):
            with it('Performs a by hourly Min between two magnitudes amb stores un ac postfix columns'):
                powpro = PowerProfile()
                powpro.load(self.erp_curve['curve'], datetime_field='utc_datetime')

                powpro.Min('ae', 'ai')

                expect(powpro.check()).to(be_true)
                for i in range(powpro.hours):
                    row = powpro[i]
                    expect(row['aeac']).to(equal(min(row['ae'], row['ai'])))

with description('PowerProfile Operators'):
    with before.all:
        self.data_path = './spec/data/'

        with open(self.data_path + 'erp_curve.json') as fp:
            self.erp_curve = json.load(fp, object_hook=datetime_parser)

    with description('Unary Operator'):
        with before.all:
            self.curve_a = PowerProfile('utc_datetime')
            self.curve_a.load(self.erp_curve['curve'])

        with context('copy'):
            with it('returns an exact copy'):
                curve_b = self.curve_a.copy()

                expect(lambda: curve_b.similar(curve_b)).not_to(raise_error)

        with context('extract'):
            with it('returns a new profile with only selected columns in a list'):
                curve_b = self.curve_a.extract(['ai_fact'])

                original_cols = list(self.curve_a.dump()[0].keys())
                expected_cols = ['utc_datetime', 'ai_fact']

                first_column = curve_b.dump()[0]
                expect(first_column).to(have_keys(*expected_cols))
                for col in list(set(original_cols) - set(expected_cols)):
                    expect(first_column).not_to(have_key(col))

            with it('raise a Value exception when field in list not in profile'):
                expect(lambda: self.curve_a.extract(['bad_field'])).to(
                    raise_error(ValueError, 'ERROR: Selected column "bad_field" does not exists in the PowerProfile')
                )
            with it('raise a Value exception when field in dict not in profile'):
                expect(lambda: self.curve_a.extract({'bad_field': 'value'})).to(
                    raise_error(ValueError, 'ERROR: Selected column "bad_field" does not exists in the PowerProfile')
                )

            with it('returns a new profile with selected columns in a dict and renamed'):
                curve_b = self.curve_a.extract({'ai_fact': 'value'})

                first_column = curve_b.dump()[0]
                expect(first_column).to(have_keys(*['utc_datetime', 'value']))
                expect(first_column).not_to(have_key('ai_fact'))

            with it('raise a Value exception when new field is not unique'):
                expect(lambda: self.curve_a.extract({'ai': 'value', 'ai_fact': 'value'})).to(
                    raise_error(ValueError, 'ERROR: Selected new name column "value" must be unique in the PowerProfile')
                )

    with description('Binary Operator'):

        with context('Extend'):
            with it('Raises Type error if not powerprofile param'):
                curve_a = PowerProfile('utc_datetime')
                curve_a.load(self.erp_curve['curve'])

                expect(
                    lambda: curve_a.extend(self.erp_curve['curve'])
                ).to(raise_error(TypeError, 'ERROR extend: Right Operand must be a PowerProfile'))

            with context('Tests profiles and'):
                with before.all:
                    self.curve_a = PowerProfile('utc_datetime')
                    self.curve_a.load(self.erp_curve['curve'])

                with it('raises a PowerProfileIncompatible with different start date'):
                    curve_b = PowerProfile('utc_datetime')
                    curve_b.load(self.erp_curve['curve'][2:])

                    expect(lambda: self.curve_a.extend(curve_b)
                    ).to(raise_error(PowerProfileIncompatible, match(r'start')))

                with it('raises a PowerProfileIncompatible with different end date'):
                    curve_b = PowerProfile('utc_datetime')
                    curve_b.load(self.erp_curve['curve'][:-2])

                    expect(lambda: self.curve_a.extend(curve_b)
                    ).to(raise_error(PowerProfileIncompatible, match(r'end')))

                with it('raises a PowerProfileIncompatible with different datetime_field'):
                    curve_b = PowerProfile('local_datetime')
                    curve_b.load(self.erp_curve['curve'])

                    expect(lambda: self.curve_a.extend(curve_b)
                    ).to(raise_error(PowerProfileIncompatible, match(r'datetime_field')))

                with it('raises a PowerProfileIncompatible with different length'):
                    curve_b = PowerProfile('utc_datetime')
                    curve_b.load([self.erp_curve['curve'][0], self.erp_curve['curve'][-1]])

                    expect(lambda: self.curve_a.extend(curve_b)
                    ).to(raise_error(PowerProfileIncompatible, match(r'hours')))


            with it('returns a new power profile with both original columns when identical'):
                curve_a = PowerProfile('utc_datetime')
                curve_a.load(self.erp_curve['curve'])

                curve_b = PowerProfile('utc_datetime')
                curve_b.load(self.erp_curve['curve'])

                curve_c = curve_a.extend(curve_b)

                expect(curve_a.hours).to(equal(curve_c.hours))
                expect(curve_b.hours).to(equal(curve_c.hours))
                expect(curve_a.start).to(equal(curve_c.start))
                expect(curve_a.end).to(equal(curve_c.end))

                extend_curve = curve_c.dump()
                first_register = extend_curve[0]
                last_register = extend_curve[-1]

                a_cols = curve_a.dump()[0].keys()
                b_cols = curve_b.dump()[0].keys()
                dfield = curve_a.datetime_field
                expected_cols = (
                        [dfield] + [a + '_left' for a in a_cols if a != dfield]
                        + [b + '_right' for b in b_cols if b != dfield]
                )

                for field in expected_cols:
                    expect(first_register.keys()).to(contain(field))

            with it('returns a new power profile with both original columns when no name col·lision'):
                curve_a = PowerProfile('utc_datetime')
                curve_a.load(self.erp_curve['curve'])

                curve_b = PowerProfile('utc_datetime')
                curve_b.load(create_test_curve(curve_a.start, curve_a.end))

                curve_c = curve_a.extend(curve_b)

                expect(curve_a.hours).to(equal(curve_c.hours))
                expect(curve_b.hours).to(equal(curve_c.hours))
                expect(curve_a.start).to(equal(curve_c.start))
                expect(curve_a.end).to(equal(curve_c.end))

                extend_curve = curve_c.dump()
                first_register = extend_curve[0]
                last_register = extend_curve[-1]

                a_cols = curve_a.dump()[0].keys()  # ae, ai, ae_fact, ai_fact, ....
                b_cols = curve_b.dump()[0].keys()  # value
                dfield = curve_a.datetime_field
                expected_cols = (
                        [dfield] + [a for a in a_cols if a != dfield]
                        + [b for b in b_cols if b != dfield]
                )

                for field in expected_cols:
                    expect(first_register.keys()).to(contain(field))

        with context('Arithmetic'):
            with before.all:
                self.data_path = './spec/data/'

                with open(self.data_path + 'erp_curve.json') as fp:
                    self.erp_curve = json.load(fp, object_hook=datetime_parser)

            with context('Scalar Multiply'):

                with it('multiplies every default value in a powerprofile by scalar integer value'):
                    curve_a = PowerProfile('utc_datetime')
                    curve_a.load(self.erp_curve['curve'])

                    new = curve_a * 2

                    expect(new).to(be_a(PowerProfile))

                    powpro_fields = curve_a.dump()[0].keys()
                    for field in powpro_fields:
                        if field in DEFAULT_DATA_FIELDS:
                            for row in new:
                                new_value = row[field]
                                old_value = curve_a[row['utc_datetime']][field]
                                expect(new_value).to(equal(old_value * 2))

                with it('multiplies one column in a powerprofile by scalar float value'):
                    curve_a = PowerProfile('utc_datetime', data_fields=['ai'])
                    curve_a.load(self.erp_curve['curve'])

                    new = curve_a * 1.5

                    expect(new).to(be_a(PowerProfile))

                    powpro_fields = curve_a.dump()[0].keys()
                    for field in powpro_fields:
                        if field in DEFAULT_DATA_FIELDS:
                            for row in new:
                                new_value = row[field]
                                old_value = curve_a[row['utc_datetime']][field]
                                if field == 'ai':
                                    expect(new_value).to(equal(old_value * 1.5))
                                else:
                                    expect(new_value).to(equal(old_value))

            with context('Profile Multiply'):
                with before.all:
                    self.curve_a = PowerProfile('utc_datetime')
                    self.curve_a.load(self.erp_curve['curve'])

                    # only test data field
                    self.curve_b = PowerProfile('utc_datetime')
                    self.curve_b.load(create_test_curve(self.curve_a.start, self.curve_a.end))

                    self.curve_c = PowerProfile('utc_datetime')
                    self.curve_c.load(create_test_curve(self.curve_a.start, self.curve_a.end))

                with it("raise a ValueError if p1 and p2 hasn't got same data fields"):

                    expect(lambda: self.curve_a + self.curve_b).to(raise_error(PowerProfileIncompatible))

                with it('multiplies value column of two profiles when p1 * p2'):
                    left = self.curve_b
                    right = self.curve_c

                    new = left * right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value * right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

                with it('multiplies every data field column of two profiles when p1 * p2'):
                    left = self.curve_a
                    right = self.curve_a
                    new = left * right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value * right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

                with it('multiplies every default data field column of two profiles when p1 * p2'):
                    left = self.curve_a
                    right = self.curve_a
                    new = left * right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value * right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

            with context('Scalar Adding'):

                with it('adds a scalar float value to every default field in a powerprofile'):
                    curve_a = PowerProfile('utc_datetime')
                    curve_a.load(self.erp_curve['curve'])

                    new = curve_a + 2.5

                    expect(new).to(be_a(PowerProfile))

                    powpro_fields = curve_a.dump()[0].keys()
                    for field in powpro_fields:
                        if field in DEFAULT_DATA_FIELDS:
                            for row in new:
                                new_value = row[field]
                                old_value = curve_a[row['utc_datetime']][field]
                                expect(new_value).to(equal(old_value + 2.5))

                with it('adds a scalar integer value to one field in a powerprofile'):
                    curve_a = PowerProfile('utc_datetime', data_fields=['ai'])
                    curve_a.load(self.erp_curve['curve'])

                    new = curve_a + 3

                    expect(new).to(be_a(PowerProfile))

                    powpro_fields = curve_a.dump()[0].keys()
                    for field in powpro_fields:
                        if field in DEFAULT_DATA_FIELDS:
                            for row in new:
                                new_value = row[field]
                                old_value = curve_a[row['utc_datetime']][field]
                                if field == 'ai':
                                    expect(new_value).to(equal(old_value + 3))
                                else:
                                    expect(new_value).to(equal(old_value))

            with context('Profile Adding'):
                with before.all:
                    self.curve_a = PowerProfile('utc_datetime')
                    self.curve_a.load(self.erp_curve['curve'])

                    # only test data field
                    self.curve_b = PowerProfile('utc_datetime')
                    self.curve_b.load(create_test_curve(self.curve_a.start, self.curve_a.end))

                    self.curve_c = PowerProfile('utc_datetime')
                    self.curve_c.load(create_test_curve(self.curve_a.start, self.curve_a.end))

                with it("raise a ValueError if p1 and p2 hasn't got same data fields"):

                    expect(lambda: self.curve_a + self.curve_b).to(raise_error(PowerProfileIncompatible))

                with it('adds value column of two profiles when p1 + p2'):
                    left = self.curve_b
                    right = self.curve_c

                    new = left + right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value + right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

                with it('adds every data field column of two profiles when p1 + p2'):
                    left = self.curve_a
                    right = self.curve_a
                    new = left + right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value + right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

                with it('adds every default data field column of two profiles when p1 + p2'):
                    left = self.curve_a
                    right = self.curve_a
                    new = left + right

                    new = left + right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value + right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

            with context('Scalar Substract'):

                with it('substacts a scalar float value to every default field in a powerprofile'):
                    curve_a = PowerProfile('utc_datetime')
                    curve_a.load(self.erp_curve['curve'])

                    new = curve_a - 0.5

                    expect(new).to(be_a(PowerProfile))

                    powpro_fields = curve_a.dump()[0].keys()
                    for field in powpro_fields:
                        if field in DEFAULT_DATA_FIELDS:
                            for row in new:
                                new_value = row[field]
                                old_value = curve_a[row['utc_datetime']][field]
                                expect(new_value).to(equal(old_value - 0.5))

                with it('substracts a scalar integer value to one field in a powerprofile'):
                    curve_a = PowerProfile('utc_datetime', data_fields=['ai'])
                    curve_a.load(self.erp_curve['curve'])

                    new = curve_a - 3

                    expect(new).to(be_a(PowerProfile))

                    powpro_fields = curve_a.dump()[0].keys()
                    for field in powpro_fields:
                        if field in DEFAULT_DATA_FIELDS:
                            for row in new:
                                new_value = row[field]
                                old_value = curve_a[row['utc_datetime']][field]
                                if field == 'ai':
                                    expect(new_value).to(equal(old_value - 3))
                                else:
                                    expect(new_value).to(equal(old_value))

            with context('Profile Substracting'):
                with before.all:
                    self.curve_a = PowerProfile('utc_datetime')
                    self.curve_a.load(self.erp_curve['curve'])

                    # only test data field
                    self.curve_b = PowerProfile('utc_datetime')
                    self.curve_b.load(create_test_curve(self.curve_a.start, self.curve_a.end))

                    self.curve_c = PowerProfile('utc_datetime')
                    self.curve_c.load(create_test_curve(self.curve_a.start, self.curve_a.end))

                with it("raise a ValueError if p1 and p2 hasn't got same data fields"):

                    expect(lambda: self.curve_a + self.curve_b).to(raise_error(PowerProfileIncompatible))

                with it('substracts value column of two profiles when p1 - p2'):
                    left = self.curve_b
                    right = self.curve_c

                    new = left - right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value - right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

                with it('substracts every data field column of two profiles when p1 - p2'):
                    left = self.curve_a
                    right = self.curve_a
                    new = left - right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value - right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

                with it('substracts every default data field column of two profiles when p1 - p2'):
                    left = self.curve_a
                    right = self.curve_a
                    new = left - right

                    expect(new).to(be_a(PowerProfile))

                    left_fields = left.dump()[0].keys()
                    new_fields = new.dump()[0].keys()
                    expect(new_fields).to(equal(left_fields))

                    for field in left_fields:
                        for row in new:
                            new_value = row[field]
                            left_value = left[row['utc_datetime']][field]
                            right_value = right[row['utc_datetime']][field]
                            if field in DEFAULT_DATA_FIELDS:
                                expect(new_value).to(equal(left_value - right_value))
                            else:
                                # no data field
                                expect(new_value).to(equal(left_value))

with description('PowerProfile Dump'):
    with before.all:
        self.data_path = './spec/data/'

        with open(self.data_path + 'erp_curve.json') as fp:
            self.erp_curve = json.load(fp, object_hook=datetime_parser)

        self.powpro = PowerProfile()
        self.powpro.load(self.erp_curve['curve'], datetime_field='utc_datetime')

    with context('to_csv'):
        with it('returns a csv file full'):
            fullcsv = self.powpro.to_csv()

            dump = read_csv(fullcsv)

            expect(len(dump)).to(equal(self.powpro.hours + 1))
            expect(list(self.powpro.curve.columns)).to(equal(dump[0]))
            header = dump[0]
            for key in header:
                for row in range(self.powpro.hours):
                    csv_value = dump[row + 1][header.index(key)]
                    powpro_value = self.powpro[row][key]
                    if isinstance(powpro_value, pd.Timestamp):
                        expect(csv_value).to(equal(powpro_value.strftime('%Y-%m-%d %H:%M:%S%z')))
                    else:
                        expect(csv_value).to(equal(str(powpro_value)))

        with it('returns a csv file without header with param header=False'):
            no_header_csv = self.powpro.to_csv(header=False)
            dump = read_csv(no_header_csv)
            header = list(self.powpro.curve.columns)

            expect(len(dump)).to(equal(self.powpro.hours))
            for key in header:
                expect(dump[0]).to_not(contain(key))

        with it('returns a csv file selected fields'):
            partial_csv = self.powpro.to_csv(['ae', 'ai'])
            dump = read_csv(partial_csv)
            header = ['utc_datetime', 'ae', 'ai']
            csv_header = dump[0]

            expect(len(dump[0])).to(equal(len(header)))

            for key in header:
                expect(csv_header).to(contain(key))

            excluded_columns = list(set(list(self.powpro.curve.columns)) - set(header))
            for key in excluded_columns:
                expect(csv_header).to_not(contain(key))

            for key in header:
                for row in range(self.powpro.hours):
                    csv_value = dump[row + 1][header.index(key)]
                    powpro_value = self.powpro[row][key]
                    if isinstance(powpro_value, pd.Timestamp):
                        expect(csv_value).to(equal(powpro_value.strftime('%Y-%m-%d %H:%M:%S%z')))
                    else:
                        expect(csv_value).to(equal(str(powpro_value)))