# -*- coding: utf-8 -*-
from expects.testing import failure
from expects import *
from powerprofile.powerprofile import PowerProfile
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_datetime
from pytz import timezone
from copy import copy
from powerprofile.exceptions import *
import json
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

            with it('with start date'):
                self.powpro.load(self.curve, start=self.start)
                expect(lambda: self.powpro.check()).not_to(raise_error)

            with it('with end date'):
                self.powpro.load(self.curve, end=self.end)
                expect(lambda: self.powpro.check()).not_to(raise_error)

            with it('with start and end date'):
                self.powpro.load(self.curve, start=self.start, end=self.end)
                expect(lambda: self.powpro.check()).not_to(raise_error)

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

    with context('Self transformation functions'):
        with context('Balance'):
            with it('Performs and by hourly Balance between two magnitudes and stores in ac postfix columns'):
                powpro = PowerProfile()
                powpro.load(self.erp_curve['curve'], datetime_field='utc_datetime')

                powpro.Balance('ae', 'ai')

                expect(powpro.check()).to(be_true)
                for i in range(powpro.hours):
                    row = powpro[i]
                    if row['ae'] >= row['ai']:
                        row['ae_bal'] = row['ae'] - row['ai']
                        row['ai_bal'] = 0.0
                    else:
                        row['ai_bal'] = row['ai'] - row['ae']
                        row['ae_bal'] = 0.0

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