# -*- coding: utf-8 -*-
from expects.testing import failure
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

        with context('with unlocalized datetimes'):
            with before.all:
                self.curve = []
                self.start = datetime(2022, 8, 1, 1, 0, 0)
                self.end = datetime(2022, 9, 1, 0, 0, 0)
                for hours in range(0, 24):
                    self.curve.append({'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours})
            with it('should localize datetimes on load'):
                powerprofile = PowerProfile()
                powerprofile.load(self.curve)
                for idx, hour in powerprofile.curve.iterrows():
                    assert hour[powerprofile.datetime_field].tzinfo is not None

    with context('fill function'):
        with context('with bad data'):
            with it('raises TypeError exception'):
                powerprofile = PowerProfile()

                expect(lambda: powerprofile.fill(['a', 'b'], datetime(2022, 8, 1, 1, 0, 0), datetime(2022, 9, 1, 0, 0, 0))).to(
                    raise_error(TypeError, "ERROR: [default_data] must be a dict")
                )

                expect(lambda: powerprofile.fill(
                    {'ae': 1.0,  "ai": 13.0}, '2020-01-31 10:00:00', datetime(2022, 9, 1, 0, 0, 0))).to(
                    raise_error(TypeError, "ERROR: [start] must be a localized datetime")
                )
                expect(lambda: powerprofile.fill(
                    {'ae': 1.0,  "ai": 13.0}, datetime(2022, 9, 1, 0, 0, 0), datetime(2022, 9, 1, 0, 0, 0))).to(
                    raise_error(TypeError, "ERROR: [start] must be a localized datetime")
                )

                expect(lambda: powerprofile.fill(
                    {'ae': 1.0,  "ai": 13.0}, LOCAL_TZ.localize(datetime(2022, 8, 1, 1, 0, 0)), '2020-01-31 10:00:00')).to(
                    raise_error(TypeError, "ERROR: [end] must be a localized datetime")
                )
                expect(lambda: powerprofile.fill(
                    {'ae': 1.0,  "ai": 13.0}, LOCAL_TZ.localize(datetime(2022, 9, 1, 0, 0, 0)), datetime(2022, 9, 1, 0, 0, 0))).to(
                    raise_error(TypeError, "ERROR: [end] must be a localized datetime")
                )

        with context('correctly'):
            with it('with correct params'):
                powerprofile = PowerProfile()

                default_data = {'name': '12345678',  'cch_bruta': True, 'bc': 0x06}
                start = LOCAL_TZ.localize(datetime(2022, 8, 1, 1, 0, 0))
                end = LOCAL_TZ.localize(datetime(2022, 9, 1, 0, 0, 0))
                powerprofile.fill(default_data, start, end)

                powerprofile.check()
                expect(powerprofile.hours).to(equal(744))
                for field, value in default_data.items():
                    expect(powerprofile[0][field]).to(equal(value))
                    expect(powerprofile[-1][field]).to(equal(value))

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
                self.curve.append(
                    {
                        'timestamp': self.start + timedelta(hours=hours),
                        'value': 100 + hours,
                        'valid':True,
                        'cch_fact': True}
                )

            self.original_curve_len = len(self.curve)

            self.data_path = './spec/data/'
            with open(self.data_path + 'curve_all.json') as fp:
                self.curve_all = json.load(fp, object_hook=datetime_parser)
            self.powpro = PowerProfile()

        with context('completeness'):

            with it('returns true when complete'):
                self.powpro.load(self.curve)

                expect(self.powpro.is_complete()[0]).to(be_true)
                expect(lambda: self.powpro.check()).not_to(raise_error)

            with it('returns false when hole'):
                curve = copy(self.curve)
                del curve[3]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len - 1))
                expect(self.powpro.is_complete()[0]).to(be_false)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileIncompleteCurve))

            with it('returns false when hole at beginning'):
                curve = copy(self.curve)
                del curve[0]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len - 1))
                expect(self.powpro.is_complete()[0]).to(be_false)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileIncompleteCurve))

            with it('returns false when hole at end'):
                curve = copy(self.curve)
                del curve[-1]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len - 1))
                expect(self.powpro.is_complete()[0]).to(be_false)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileIncompleteCurve))

            with it('returns false when complete but duplicated hours'):
                curve = copy(self.curve)
                curve.append(curve[-1])
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.is_complete()[0]).to(be_false)
                expect(self.powpro.is_complete()[1]).to(equal(curve[0]['timestamp']))
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileDuplicatedTimes))

            with it('returns false when incomplete but duplicated hours complete number of hours'):
                curve = copy(self.curve[1:])
                curve.append(curve[-1])
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.is_complete()[0]).to(be_false)
                expect(self.powpro.is_complete()[1]).to(equal(self.curve[0]['timestamp']))
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileDuplicatedTimes))

        with context('duplicated hours'):

            with it('returns true when not duplicates'):
                self.powpro.load(self.curve)

                expect(self.powpro.has_duplicates()[0]).to(be_false)
                expect(lambda: self.powpro.check()).not_to(raise_error)

            with it('returns false when duplicates extra hour'):
                curve = copy(self.curve)
                curve.append(self.curve[3])
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len + 1))
                expect(self.powpro.has_duplicates()[0]).to(be_true)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileDuplicatedTimes))

            with it('returns false when duplicates and correct length'):
                curve = copy(self.curve)
                curve.append(self.curve[3])
                del curve[0]
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.hours).to(equal(self.original_curve_len))
                expect(self.powpro.has_duplicates()[0]).to(be_true)
                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileDuplicatedTimes))

        with context('curve fixed'):
            with it('returns true valid and cch_fact is true on all registers'):
                self.powpro.load(self.curve)

                expect(self.powpro.has_duplicates()[0]).to(be_false)
                expect(lambda: self.powpro.is_fixed(['valid', 'cch_fact'])).not_to(raise_error)

            with it('returns false when one register is not valid'):
                curve = deepcopy(self.curve)
                curve[3]['valid'] = False
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.is_fixed(['valid', 'cch_fact'])).to(be_false)

            with it('returns false when one register is not cch_fact'):
                curve = deepcopy(self.curve)
                curve[3]['cch_fact'] = False
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.is_fixed(['valid', 'cch_fact'])).to(be_false)

            with it('returns false when one register is neither cch_fact nor valid'):
                curve = deepcopy(self.curve)
                curve[3]['cch_fact'] = curve[3]['cch_fact'] = False
                self.powpro.load(curve, self.start, self.end)

                expect(self.powpro.is_fixed(['valid', 'cch_fact'])).to(be_false)

            with it('returns PowerProfileMissingField Exception'):
                self.powpro.load(self.curve, self.start, self.end)

                expect(lambda: self.powpro.is_fixed(['missing_field'])).to(raise_error(PowerProfileMissingField))

        with context('curve positive'):
            with it('returns true when all columns > 0'):
                self.powpro.load(self.curve_all['curve'], datetime_field='utc_datetime')

                expect(self.powpro.is_positive()).to(be_true)

            with it('returns PowerProfileNegativeCurve Exception when one ai is negative'):
                curve = deepcopy(self.curve_all['curve'])
                curve[0]['ai'] = -3
                self.powpro.load(curve, datetime_field='utc_datetime')

                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileNegativeCurve))

            with it('returns PowerProfileNegativeCurve Exception when one ai_fact is negative'):
                curve = deepcopy(self.curve_all['curve'])
                curve[0]['ai_fact'] = -3
                self.powpro.load(curve, datetime_field='utc_datetime')

                expect(lambda: self.powpro.check()).to(raise_error(PowerProfileNegativeCurve))

            with it('returns PowerProfileNegativeCurve Exception when any measure is negative'):
                for field in DEFAULT_DATA_FIELDS:
                    curve = deepcopy(self.curve_all['curve'])
                    curve[0][field] = -4
                    self.powpro.load(curve, datetime_field='utc_datetime')

                    expect(lambda: self.powpro.check()).to(raise_error(PowerProfileNegativeCurve))

        with context('complete subcurve'):
            with before.all:
                self.curve_subcurve_testing = []
                self.start = LOCAL_TZ.localize(datetime(2020, 3, 11, 1, 0, 0))
                self.end = LOCAL_TZ.localize(datetime(2020, 3, 13, 0, 0, 0))
                for hours in range(0, 48):
                    self.curve_subcurve_testing.append(
                        {'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours}
                    )
                self.powpro_subcurve_testing = PowerProfile()

                self.curve_subcurve_testing2 = []
                for hours in range(0, 43):
                    self.curve_subcurve_testing2.append(
                        {'timestamp': self.start + timedelta(hours=hours), 'value': 100 + hours}
                    )
                for hours in range(43, 48):
                    self.curve_subcurve_testing2.append(
                        {'timestamp': self.start + timedelta(hours=hours - 1), 'value': 100 + hours}
                    )
                self.powpro_subcurve_testing2 = PowerProfile()

                self.powpro_subcurve_testing3 = PowerProfile()

            with it('returns first complete part of curve if there are gaps'):
                curve = copy(self.curve_subcurve_testing)
                del curve[-1]
                self.powpro_subcurve_testing.load(curve, self.start, self.end)
                pp = self.powpro_subcurve_testing.get_complete_daily_subcurve()
                expect(len(pp.curve)).to(equal(24))
                expect(pp.curve['timestamp'].max()).to(equal(self.curve_subcurve_testing[23]['timestamp']))

            with it('returns first complete part of curve if there are duplicated hours'):
                curve = copy(self.curve_subcurve_testing2)
                del curve[-1]
                self.powpro_subcurve_testing2.load(curve, self.start, self.end)
                pp = self.powpro_subcurve_testing2.get_complete_daily_subcurve()
                expect(len(pp.curve)).to(equal(24))
                expect(pp.curve['timestamp'].max()).to(equal(self.curve_subcurve_testing2[23]['timestamp']))

            with it('returns empty PowerProfile if first hour is lost'):
                curve = copy(self.curve_subcurve_testing)
                del curve[0]
                self.powpro_subcurve_testing3.load(curve, self.start, self.end)
                pp = self.powpro_subcurve_testing3.get_complete_daily_subcurve()
                expect((pp.curve)).to(equal(None))

        with context('complete season curve'):
            with before.all:
                self.curve_season_testing = []
                self.start = LOCAL_TZ.localize(datetime(2020, 1, 1, 1, 0, 0))
                self.end = LOCAL_TZ.localize(datetime(2022, 1, 1, 0, 0, 0))

                num_hours = int((self.end - self.start).total_seconds() / 3600) + 1
                for hours in range(0, num_hours):
                    self.curve_season_testing.append(
                        {'timestamp': LOCAL_TZ.normalize(self.start + timedelta(hours=hours)), 'value': 100 + hours}
                    )
                self.powpro_season_testing = PowerProfile(datetime_field='timestamp')
                self.powpro_season_testing.load(self.curve_season_testing)

                self.total_registers = self.powpro_season_testing.samples

                winter_pp = self.powpro_season_testing.get_winter_curve()
                # Abril no és hivern
                expect(LOCAL_TZ.localize(datetime(2020, 4, 1, 0)) not in winter_pp.curve.timestamp).to(be_true)
                # Novembre és hivern
                expect(LOCAL_TZ.localize(datetime(2020, 11, 1, 0)) not in winter_pp.curve.timestamp).to(be_true)

                self.winter_registers = winter_pp.samples
                expect(self.winter_registers).to(be_below(self.total_registers))

                summer_pp = self.powpro_season_testing.get_summer_curve()
                # Abril és estiu"
                expect(LOCAL_TZ.localize(datetime(2020, 4, 1, 0)) in summer_pp.curve.timestamp).to(be_true)
                # Novembre no és estiu
                expect(LOCAL_TZ.localize(datetime(2020, 11, 1, 0)) not in summer_pp.curve.timestamp).to(be_true)

                self.summer_registers = summer_pp.samples
                expect(self.summer_registers).to(be_below(self.total_registers))


                expect(self.summer_registers + self.winter_registers).to(be_equal(self.total_registers))



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

        with context('ApplyLbtLosses'):
            with it('Performs an application of LBT losses and store it in sufix columns'):
                powpro = PowerProfile()
                powpro.load(self.erp_curve['curve'], datetime_field='utc_datetime')

                # Prepare _fix columns
                powpro.curve['ai_fix'] = powpro.curve['ai']
                powpro.curve['ae_fix'] = powpro.curve['ae']

                trafo = 50  # kVA
                losses = 0.04  # %
                sufix = '_fix'  # 'ae_fix' and 'ai_fix'
                powpro.ApplyLbtLosses(trafo, losses, sufix)

                expect(powpro.check()).to(be_true)
                for i in range(powpro.hours):
                    row = powpro[i]
                    # Max error allower is 1000 Wh (1 kWh)
                    assert abs(round(row['ai_fix'], 1) - round((row['ai'] * (1 + losses) + (10 * trafo)), 1)) <= 1000.0
                    assert abs(round(row['ae_fix'], 1) - round((row['ae'] * (1 - losses)), 1)) <= 1000.0

        with context('Dragg'):
            with it('performs a dragging operation through float values of specified magnitudes on curve'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')
                real_ai_sum = powpro.curve['ai'].sum()
                real_ae_sum = powpro.curve['ae'].sum()

                powpro.drag(['ai', 'ae'])
                dragged_ai_sum = powpro.curve['ai'].sum()
                dragged_ae_sum = powpro.curve['ae'].sum()

                # Max error allower is 1000 Wh (1 kWh)
                assert abs(real_ai_sum - dragged_ai_sum) <= 1000.0
                assert abs(real_ae_sum - dragged_ae_sum) <= 1000.0

        with context('min'):
            with it('performs a min operation through float values of specified magnitude on curve'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                min_ai = powpro.min('ai')
                min_ai_timestamp = powpro.min('ai', 'timestamp')
                min_ae = powpro.min('ae')
                min_ae_timestamp = powpro.min('ae', 'timestamp')

                json_min_ai = min(curve_all['curve'], key=lambda x: x['ai'])
                json_min_ae = min(curve_all['curve'], key=lambda x: x['ae'])

                expect(min_ai).to(equal(json_min_ai['ai']))
                expect(min_ai_timestamp).to(
                    equal(LOCAL_TZ.localize(datetime(2022, 2, 1, 11)))
                )
                expect(min_ae).to(equal(json_min_ae['ae']))
                expect(min_ae_timestamp).to(
                    equal(LOCAL_TZ.localize(datetime(2022, 2, 1, 1)))
                )
            with it('returns error if magn is invalid'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                expect(lambda: powpro.min('invalid_magn')).to(
                    raise_error(ValueError, 'ERROR: [magn] is not a valid parameter, given magn: invalid_magn')
                )

        with context('max'):
            with it('Performs a max operation through float values of specified magnitude on curve'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                max_ai = powpro.max('ai')
                max_ai_timestamp = powpro.max('ai', 'timestamp')
                max_ae = powpro.max('ae')
                max_ae_timestamp = powpro.max('ae', 'timestamp')


                json_max_ai = max(curve_all['curve'], key=lambda x: x['ai'])
                json_max_ae = max(curve_all['curve'], key=lambda x: x['ae'])

                expect(max_ai).to(equal(json_max_ai['ai']))
                expect(max_ai_timestamp).to(
                    equal(LOCAL_TZ.localize(datetime(2022, 2, 12, 23)))
                )
                expect(max_ae).to(equal(json_max_ae['ae']))
                expect(max_ae_timestamp).to(
                    equal(LOCAL_TZ.localize(datetime(2022, 2, 22, 14)))
                )

            with it('returns error if magn is invalid'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                expect(lambda: powpro.max('invalid_magn')).to(
                    raise_error(ValueError, 'ERROR: [magn] is not a valid parameter, given magn: invalid_magn')
                )

        with context('avg'):
            with it('performs a avg operation through float values of specified magnitude on curve'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                avg_ai = powpro.avg('ai')
                avg_ae = powpro.avg('ae')

                json_sum_ai = sum([cur['ai'] for cur in curve_all['curve']])
                json_avg_ai = json_sum_ai / len(curve_all['curve'])

                json_sum_ae = sum([cur['ae'] for cur in curve_all['curve']])
                json_avg_ae = json_sum_ae / len(curve_all['curve'])

                expect(round(avg_ai, 6)).to(equal(round(json_avg_ai, 6)))
                expect(round(avg_ae, 6)).to(equal(round(json_avg_ae, 6)))

            with it('returns error if magn is invalid'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                expect(lambda: powpro.avg('invalid_magn')).to(
                    raise_error(ValueError, 'ERROR: [magn] is not a valid parameter, given magn: invalid_magn')
                )

        with context('get_n_rows'):
            with it('performs a query and gets the newest row if repeated'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                max_latest_ae = powpro.get_n_rows(['ae'], 'last', 1)
                max_first_ae = powpro.get_n_rows(['ae'], 'first', 1)

                latest_date = powpro.convert_numpydate_to_datetime(max_latest_ae['timestamp'].values[0])
                first_date = powpro.convert_numpydate_to_datetime(max_first_ae['timestamp'].values[0])
                expect(latest_date > first_date).to(equal(True))

            with it('performs a query and gets the oldest row if repeated'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                max_latest_ae = powpro.get_n_rows(['ae'], 'last', 1, 'asc')
                max_first_ae = powpro.get_n_rows(['ae'], 'first', 1, 'asc')

                latest_date = powpro.convert_numpydate_to_datetime(max_latest_ae['timestamp'].values[0])
                first_date = powpro.convert_numpydate_to_datetime(max_first_ae['timestamp'].values[0])
                expect(latest_date > first_date).to(equal(True))

            with it('returns error if keep is invalid'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                expect(lambda: powpro.get_n_rows(['ae'], 'invalid_keep', 1, 'asc')).to(
                    raise_error(ValueError, "ERROR: [keep] is not a valid parameter, given keep: invalid_keep."
                             "Valid keep options are 'first', 'last, 'all'")
                )

            with it('returns error if order is invalid'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                expect(lambda: powpro.get_n_rows(['ae'], 'first', 1, 'invalid_order')).to(
                    raise_error(ValueError, "ERROR: [order] is not a valid parameter, given keep: invalid_order."
                             "Valid keep options are 'asc', 'desc'")
                )

            with it('returns error if cols type is invalid'):
                data_path = './spec/data/'
                with open(data_path + 'demo_ac_curve.json') as fp:
                    curve_all = json.load(fp, object_hook=datetime_parser)

                powpro = PowerProfile()
                powpro.load(curve_all['curve'], datetime_field='timestamp')

                expect(lambda: powpro.get_n_rows('ae', 'first', 1)).to(
                    raise_error(TypeError, "ERROR: [cols] has to be a list, given keep: ae.")
                )

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

                    try:
                        self.curve_a.extend(curve_b)
                    except Exception as e:
                        expect(str(e)).to(contain('start'))

                with it('raises a PowerProfileIncompatible with different end date'):
                    curve_b = PowerProfile('utc_datetime')
                    curve_b.load(self.erp_curve['curve'][:-2])

                    try:
                        self.curve_a.extend(curve_b)
                    except Exception as e:
                        expect(str(e)).to(contain('end'))

                with it('raises a PowerProfileIncompatible with different datetime_field'):
                    curve_b = PowerProfile('local_datetime')
                    curve_b.load(self.erp_curve['curve'])

                    try:
                        self.curve_a.extend(curve_b)
                    except Exception as e:
                        expect(str(e)).to(contain('datetime_field'))

                with it('raises a PowerProfileIncompatible with different length'):
                    curve_b = PowerProfile('utc_datetime')
                    curve_b.load([self.erp_curve['curve'][0], self.erp_curve['curve'][-1]])

                    try:
                        self.curve_a.extend(curve_b)
                    except Exception as e:
                        expect(str(e)).to(contain('hours'))



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

        ## End Extend tests

        with context('Append'):
            with it('Raises Type error if not powerprofile param'):
                curve_a = PowerProfile('utc_datetime')
                curve_a.load(self.erp_curve['curve'])

                expect(
                    lambda: curve_a.append(self.erp_curve['curve'])
                ).to(raise_error(TypeError, 'ERROR append: Appended Profile must be a PowerProfile'))

            with context('Tests profiles and'):
                with before.all:
                    self.curve_a = PowerProfile('utc_datetime')
                    self.curve_a.load(self.erp_curve['curve'])

                with it('raises a PowerProfileIncompatible with different profile type'):
                    curve_b = PowerProfileQh('utc_datetime')
                    curve_b.load(self.erp_curve['curve'])

                    try:
                        self.curve_a.append(curve_b)
                    except Exception as e:
                        expect(str(e)).to(contain('different profile type'))

                with it('raises a PowerProfileIncompatible with different datetime field'):
                    curve_b = PowerProfile('local_datetime')
                    curve_b.load(self.erp_curve['curve'])

                    try:
                        self.curve_a.append(curve_b)
                    except Exception as e:
                        expect(str(e)).to(contain('datetime field'))

                with it('raises a PowerProfileIncompatible with different fields'):
                    curve_b = PowerProfile('utc_datetime')
                    curve_b.load(self.erp_curve['curve'])
                    curve_c = curve_b.extract(['ai', 'ae'])

                    try:
                        self.curve_a.append(curve_c)
                    except PowerProfileIncompatible as e:
                        expect(str(e)).to(contain('ai_fact'))

            with it('returns a new power profile with both data'):
                curve_a = PowerProfile('utc_datetime')
                curve_a.load(self.erp_curve['curve'])

                curve_b = PowerProfile('utc_datetime')
                curve_b.load(self.erp_curve['curve'])

                curve_c = curve_a.append(curve_b)

                expect(curve_a.hours * 2).to(equal(curve_c.hours))

                expect(curve_c.start).to(equal(curve_a.start))
                expect(curve_c.end).to(equal(curve_b.end))

        ## End Append tests

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