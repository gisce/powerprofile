# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime, timedelta
from pytz import timezone
import pandas as pd
import copy
from .exceptions import *
from .utils import Dragger
import math
try:
    # Python 2
    from cStringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO


TIMEZONE = timezone('Europe/Madrid')
UTC_TIMEZONE = timezone('UTC')


DEFAULT_DATA_FIELDS_NO_FACT = ['ae', 'ai', 'r1', 'r2', 'r3', 'r4']
DEFAULT_DATA_FIELDS = DEFAULT_DATA_FIELDS_NO_FACT + [f + '_fact' for f in DEFAULT_DATA_FIELDS_NO_FACT] + [f + '_fix' for f in DEFAULT_DATA_FIELDS_NO_FACT] + ['value']


class PowerProfile():

    SAMPLING_INTERVAL = 3600

    def __init__(self, datetime_field='timestamp', data_fields=DEFAULT_DATA_FIELDS):

        self.start = None
        self.end = None
        self.curve = None
        self.datetime_field = datetime_field
        self.data_fields = data_fields

    def load(self, data, start=None, end=None, datetime_field=None, data_fields=None):
        if not isinstance(data, (list, tuple)):
            raise TypeError("ERROR: [data] must be a list of dicts ordered by timestamp")

        if start and not isinstance(start, datetime):
            raise TypeError("ERROR: [start] must be a localized datetime")

        if end and not isinstance(end, datetime):
            raise TypeError("ERROR: [end] must be a localized datetime")

        if datetime_field is not None:
            self.datetime_field = datetime_field

        if data and not data[0].get(self.datetime_field, False):
            raise TypeError("ERROR: No timestamp field. Use datetime_field option to set curve datetime field")

        self.curve = pd.DataFrame(data)
        self.curve.sort_values(by=self.datetime_field, inplace=True)
        self.curve.reset_index(inplace=True, drop=True)
        # Ensure timestamp field is localized
        self.curve[self.datetime_field] = self.curve.apply(lambda row: self.ensure_localized_dt(row), axis=1)

        if start:
            self.start = start
        else:
            self.start = self.curve[self.datetime_field].min()

        if end:
            self.end = end
        else:
            self.end = self.curve[self.datetime_field].max()

        if data_fields is not None:
            self.data_fields = data_fields
        else:
            auto_data_fields = []
            loaded_data_fields = data[0].keys()
            for field in DEFAULT_DATA_FIELDS:
                if field in loaded_data_fields and field in self.data_fields:
                    auto_data_fields.append(field)
            self.data_fields = auto_data_fields

    def fill(self, default_data, start, end):
        '''
        Fills curve with default data
        :param data: dict with field and default value, ie: {'ai': 0, 'ae': 0, 'cch_bruta': False}
        '''
        if not isinstance(default_data, dict):
            raise TypeError("ERROR: [default_data] must be a dict")

        if not isinstance(start, datetime) or not start.tzinfo:
            raise TypeError("ERROR: [start] must be a localized datetime")

        if not isinstance(end, datetime) or not end.tzinfo:
            raise TypeError("ERROR: [end] must be a localized datetime")

        self.start = start
        self.end = end

        data = []
        sample_counter = 0
        ts = copy.copy(self.start)
        while self.end > ts:
            append_data = {}
            ts = TIMEZONE.normalize(self.start + timedelta(seconds=sample_counter * self.SAMPLING_INTERVAL))
            append_data[self.datetime_field] = ts
            append_data.update(default_data)
            data.append(append_data)

            sample_counter += 1

        self.load(data)

    def ensure_localized_dt(self, row):
        dt = row[self.datetime_field]
        if dt.tzinfo is None:
            return TIMEZONE.localize(dt)
        else:
            return dt

    def dump(self):

        data = self.curve.to_dict(orient='records')

        return data

    @property
    def samples(self):
        if self.SAMPLING_INTERVAL == 900:
            return self.quart_hours
        else:
            return self.hours

    @property
    def hours(self):
        return int(self.curve.count()[self.datetime_field] / (self.SAMPLING_INTERVAL / 3600))

    @property
    def quart_hours(self):
        return int(self.curve.count()[self.datetime_field] / (self.SAMPLING_INTERVAL / 900))

    @property
    def unique_samples(self):
        return int(len(self.curve[self.datetime_field].unique()))

    def is_complete_counter(self, counter):
        ''' Checks completeness of curve '''
        start = self.start
        if self.start.tzinfo is None or self.start.tzinfo.utcoffset(self.start) is None:
            start = TIMEZONE.localize(self.start)
        end = self.end
        if self.end.tzinfo is None or self.end.tzinfo.utcoffset(self.end) is None:
            end = TIMEZONE.localize(self.end)

        samples = ((end - start).total_seconds() + self.SAMPLING_INTERVAL) / self.SAMPLING_INTERVAL
        if counter != samples or counter != self.unique_samples:
            ids = set(self.curve[self.datetime_field])
            dt = start
            df_hours = set([TIMEZONE.normalize(dt + timedelta(seconds=x * self.SAMPLING_INTERVAL)) for x in range(0, int(samples))])
            not_found = sorted(list(df_hours - ids))
            if len(not_found):
                first_not_found = not_found[0]
            else:
                first_not_found = dt
            return False, first_not_found
        return True, None

    def is_complete(self):
        return self.is_complete_counter(self.hours)

    def is_fixed(self, fields=['cch_fact', 'valid']):
        """
        Given a list of fields, check all values are True in every register
        :param fields []:
        :return: list
        """
        for field in fields:
            try:
                data = self.curve.loc[self.curve[field] == False]
                if len(data) > 0:
                    return False
            except KeyError as e:
                raise PowerProfileMissingField(field)
        return True

    def has_duplicates_counter(self, counter):
        ''' Checks for duplicated hours'''
        uniques = len(self.curve[self.datetime_field].unique())
        if uniques != counter:
            ids = self.curve[self.datetime_field]
            first_occurrence = self.curve[ids.isin(ids[ids.duplicated()])][self.datetime_field].min()
            return True, first_occurrence
        return False, None

    def has_duplicates(self):
        return self.has_duplicates_counter(self.hours)

    def is_positive(self, fields=DEFAULT_DATA_FIELDS):
        """
        Checks if the curve does not have any negative value
        :param fields: list
        :return: boolean
        """
        curve_fields = list(self.curve.columns)
        common_fields = list(set(fields).intersection(curve_fields))

        for field in common_fields:
            data = self.curve.loc[self.curve[field] < 0]
            if len(data) > 0:
                return False
        return True

    def check(self):
        '''Tests curve validity'''
        has_duplicates, first_duplicated = self.has_duplicates()
        if has_duplicates:
            raise PowerProfileDuplicatedTimes(message=first_duplicated)
        is_complete, first_not_found = self.is_complete()
        if not is_complete:
            raise PowerProfileIncompleteCurve(message=first_not_found)
        if not self.is_positive():
            raise PowerProfileNegativeCurve
        return True

    def __getitem__(self, item):
        if isinstance(item, int):
            #interger position
            res = self.curve.iloc[item]
            return dict(res)
        elif isinstance(item, slice):
            res = self.curve.iloc[item]
            #interger slice [a:b]
            # test bounds
            self.curve.iloc[item.start or 0]  # Python 3 returns None istead of 0 when empty
            self.curve.iloc[item.stop or -1]  # Python 3 returns None instead of -1 when empty
            powpro = self.__class__()
            powpro.curve = res
            powpro.start = res.iloc[0][self.datetime_field]
            powpro.end = res.iloc[-1][self.datetime_field]
            return powpro
        elif isinstance(item, datetime):
            if not item.tzinfo:
                raise TypeError('Datetime must be a localized datetime')

            res = self.curve.loc[self.curve[self.datetime_field] == item]
            return dict(res.iloc[0])

    # Aggregations
    def sum(self, magns):
        """
        Sum of every value in every row of the curve
        :param magns: magnitudes
        :return: dict a key for every magnitude in magns dict
        """
        totals = self.curve[magns].sum()
        res = {}
        for magn in magns:
            res[magn] = totals[magn]
        return res

    def max(self, magn, ret='value'):
        """
        Returns max value of given magnitude of the curve
        :param magn: magnitude value
        :param ret: value or timestamp of maximum
        :return: max magnitude value
        """
        if self._check_magn_is_valid(magn):
            if ret == 'value':
                return self.curve[magn].max()
            elif ret == 'timestamp':
                idx_max = self.curve[magn].idxmax()
                return self[idx_max][self.datetime_field]

    def min(self, magn, ret='value', magn_0=None, force_magn_gt_0=False):
        """
        Returns min value of given magnitude of the curve
        :param magn: magnitude value
        :param magn_0: magnitude that when specified returns only the min value where **magn_0** is 0.
        :param force_magn_gt_0: flag that forces **magn** to be > 0
        :param ret: value or timestamp of minimum
        :return: min magnitude value
        """
        if self._check_magn_is_valid(magn):
            filtered = self.curve.copy()

            if force_magn_gt_0:
                filtered = filtered[self.curve[magn] > 0]

            if magn_0 is not None and self._check_magn_is_valid(magn_0):
                filtered = filtered[self.curve[magn_0] == 0]

            if not filtered.empty:
                if ret == 'value':
                    return filtered[magn].min()
                elif ret == 'timestamp':
                    idx_min = filtered[magn].idxmin()
                    return self[idx_min][self.datetime_field]

        return False

    def avg(self, magn):
        """
        Returns avg value of given magnitude of the curve
        :param magn: magnitude value
        :return: avg magnitude value
        """
        if self._check_magn_is_valid(magn):
            return self.curve[magn].mean()


    def std(self, magn):
        """
        Returns std value of given magnitude of the curve
        :param magn: magnitude value
        :return: std magnitude value
        """
        if self._check_magn_is_valid(magn):
            return self.curve[magn].std()

    # Transformations
    def Balance(self, magn1='ai', magn2='ae', sufix='bal'):
        """
        Balance two magnitude row by row. It perfoms the difference between both magnitudes and stores 0.0 in the
        little one and the difference in the big one. The result is stored in two new fields with the same name of
        selected magnitudes with selected postfix
        :param magn1: magnitude 1. 'ae' as default
        :param magn2: magnitude 2. 'ai' as default
        :param sufix: postfix of new fields 'bal' as default
        :return:
        """
        def balance(pos, neg):
            res = pos - neg
            if res > 0.0:
                return res
            else:
                return 0.0

        self.curve[magn1 + sufix] = self.curve.apply(lambda row: balance(row[magn1], row[magn2]), axis=1)
        self.curve[magn2 + sufix] = self.curve.apply(lambda row: balance(row[magn2], row[magn1]), axis=1)

    def ApplyLbtLosses(self, trafo, losses, sufix='_fix'):
        """
        Adds losses and trafo charge to consumption. Subs losses to generation. Curve is expressed in Wh.
        :param trafo: float (expressed in kVA)
        :param losses: float (usually, 0.04)
        :param sufix: str (magn where to apply losses, usually '_fix')
        :return:
        """
        def elevate(kva, losses, value):
            return round(value * (1 + losses), 2) + round(10 * kva, 2)

        def descend(losses, value):
            return round(value * (1 - losses), 2)

        self.curve['ai' + sufix] = self.curve.apply(lambda row: elevate(trafo, losses, row['ai' + sufix]), axis=1)
        self.curve['ae' + sufix] = self.curve.apply(lambda row: descend(losses, row['ae' + sufix]), axis=1)

    def drag(self, magns, drag_key=None):
        """
        Allows to drag the decimal values for specified magns on current curve
        :param magns: list of str (e.g. ['ai_fix', 'ae_fix'])
        :param drag_key: str (e.g. 'period', None by default)
        :return:
        """
        for magn in magns:
            draggers = Dragger()
            # Dragg field is specified and exists in curve
            if drag_key is not None and drag_key in self.curve:
                self.curve[magn] = self.curve.apply(lambda row: draggers.drag(round(row[magn] / (1000 / row.get('magn', 1)), 6),
                                                                              key=row[drag_key]), axis=1)
            else:
                self.curve[magn] = self.curve.apply(lambda row: draggers.drag(round(row[magn] / (1000 / row.get('magn', 1)), 6)), axis=1)
            self.curve[magn] = self.curve.apply(lambda row: row[magn] * (1000 / row.get('magn', 1)), axis=1)

    def Min(self, magn1='ae', magn2='ai', sufix='ac'):
        """
        Allows easy AUTOCONSUMED curve from a curve with both exported and imported energy
        Makes a positive always difference between two magnitudes row by row. It performs the difference between both
        magnitudes:
         * if positive, returns the value
         * if negative or zero, returns first term (magn1)
        of the magn1 selected magnitude with selected postfix
        :param magn1: magnitude 1. 'ae' as default
        :param magn2: magnitude 2. 'ai' as default
        :param sufix: postfix of new fields 'ac' as default
        :return:
        """
        self.curve[magn1 + sufix] = self.curve.apply(lambda row: min(row[magn1], row[magn2]), axis=1)

    # Operators
    def check_data_fields(self, right):
        if len(self.data_fields) != len(right.data_fields):
            raise PowerProfileIncompatible('ERROR: right data fields "{}" are not the same: {}'.format(
                self.data_fields, right.data_fields)
            )
        for field in self.data_fields:
            if field not in right.data_fields:
                raise PowerProfileIncompatible('ERROR: right profile does not contains field "{}": {}'.format(
                    field, right.data_fields)
                )
        return True

    # Binary
    def similar(self, right, data_fields=False):
        """Ensures two PowerProfiles are "compatible", that is:
            * same start date
            * same end date
            * same datetime_field
            * same length

        :param right:
        :param data_fields: Test also same data_fields if True
        :return: True if ok , raises PowerProfileIncompatible instead
        """
        for field in ['start', 'end', 'datetime_field', 'hours']:
            if getattr(right, field) != getattr(self, field):
                raise PowerProfileIncompatible('ERROR: right "{}" attribute {} is not equal: {}'.format(
                    field, getattr(right, field), getattr(self, field)))

        if data_fields:
            self.check_data_fields(right)

        return True

    def __operate(self, right, op='mul'):
        new = self.copy()
        scalar = False
        if isinstance(right, (int, float)):
            # scalar
            scalar = True
        else:
            self.similar(right, data_fields=True)

        if op in ['mul', 'rmul']:
            for field in self.data_fields:
                if scalar:
                    new.curve[field] = self.curve[field] * right
                else:
                    new.curve[field] = self.curve[field] * right.curve[field]
        elif op in ['add', 'radd']:
            for field in self.data_fields:
                if scalar:
                    new.curve[field] = self.curve[field] + right
                else:
                    new.curve[field] = self.curve[field] + right.curve[field]

        elif op in ['sub']:
            for field in self.data_fields:
                if scalar:
                    new.curve[field] = self.curve[field] - right
                else:
                    new.curve[field] = self.curve[field] - right.curve[field]
        elif op in ['rsub']:
            for field in self.data_fields:
                if scalar:
                    new.curve[field] = - (self.curve[field] - right)
                else:
                    new.curve[field] = right.curve[field] - self.curve[field]

        return new

    def __mul__(self, other, op='mul'):
        return self.__operate(other, op)

    def __rmul__(self, other, op='rmul'):
        return self.__operate(other, op)

    def __add__(self, other, op='add'):
        return self.__operate(other, op)

    def __radd__(self, other, op='radd'):
        return self.__operate(other, op)

    def __sub__(self, other, op='sub'):
        return self.__operate(other, op)

    def __rsub__(self, other, op='rsub'):
        return self.__operate(other, op)

    def extend(self, right):
        ''' Add right curve columns to current curve and return a new curve. It adds _left and _right suffix
        on every column depending on origin'''
        if not isinstance(right, PowerProfile):
            raise TypeError('ERROR extend: Right Operand must be a PowerProfile')

        self.similar(right)

        new = self.copy()
        new.curve = self.curve.merge(
            right.curve, how='inner', on=self.datetime_field, suffixes=('_left', '_right'), validate='one_to_one'
        )

        return new

    def append(self, new_profile):
        '''Appends data to to current curve. Usefull to fill gaps or strech the profile'''
        if not isinstance(new_profile, PowerProfile):
            raise TypeError('ERROR append: Appended Profile must be a PowerProfile')

        #if type(self) is not type(new_profile):
        if self.SAMPLING_INTERVAL != new_profile.SAMPLING_INTERVAL:
            raise PowerProfileIncompatible(
                "ERROR: Can't append profiles of different profile type: {} != {}".format(self.__class__, new_profile.__class__)
            )

        if self.datetime_field != new_profile.datetime_field:
            raise PowerProfileIncompatible(
                "ERROR: Can't append profiles of different datetime field: {} != {}".format(
                    self.datetime_field , new_profile.datetime_field
                )
            )

        self.check_data_fields(new_profile)

        new_curve = self.copy()

        new_curve.curve = pd.concat([new_curve.curve, new_profile.curve])
        new_curve.curve.sort_values(by=new_curve.datetime_field, inplace=True)
        new_curve.curve.reset_index(inplace=True, drop=True)
        new_curve.start = new_curve.curve[new_curve.datetime_field].min()
        new_curve.end = new_curve.curve[new_curve.datetime_field].max()

        return new_curve

    # Unary
    def copy(self):
        """
        Returns an identical copy of the same profile
        :return: PowerProfile Object
        """
        new = self.__class__(self.datetime_field)
        new.start = self.start
        new.end = self.end
        new.curve = copy.copy(self.curve)
        new.data_fields = copy.copy(self.data_fields)

        return new

    def extract(self, cols):
        """
        Returns a new profile with only selected columns
        When cols is a list, the new profile contains only datetime field and selected columns in the list
            ['col1', 'col2']
        When cols is a dict, also renames selected columns (key) to the new value:
            {'orig_col1': 'new_col_name1', 'orig_col2': 'new_col_name2}
        Raise a Value Error when selected column is not in the current profile
        :return: The new Profile
        """
        new = self.copy()
        if isinstance(cols, dict):
            selected_cols = list(cols.keys())
        else:
            selected_cols = cols

        current_cols = self.curve.head()
        # test cols
        for col in selected_cols:
            if col not in current_cols:
                raise ValueError('ERROR: Selected column "{}" does not exists in the PowerProfile'.format(col))

        final_cols = [self.datetime_field] + selected_cols
        for col in current_cols:
            if col not in final_cols:
                del new.curve[col]

        # field translation
        if isinstance(cols, dict):
            # test new name exists
            final_trans_cols = list(cols.values())
            for col in final_trans_cols:
                if final_trans_cols.count(col) > 1:
                    raise ValueError('ERROR: Selected new name column "{}" must be unique in the PowerProfile'.format(col))

            new.curve.rename(columns=cols, inplace=True)

            final_cols = final_trans_cols[:]

        new_data_fields = [x for x in final_cols if x != self.datetime_field]
        new.data_fields = new_data_fields

        return new

    def get_summer_curve(self):
        return self.get_season_curve(dst=True)

    def get_winter_curve(self):
        return self.get_season_curve(dst=False)

    def get_season_curve(self, dst=True):
        """
        Returns a new partial profile with only the summer registers (dst=True) or the winter registers (false) using
        DST change time of local timezone
        :param dst: boolean True(summer) False(winter)
        :return: new profile
        """
        df = self.curve.copy()
        df['dst'] = df.apply(
            lambda row: True if TIMEZONE.normalize(row[self.datetime_field]).dst() else False, axis=1
        )
        df = df[df['dst'] == dst]

        res = self.__class__(datetime_field=self.datetime_field)
        data = df.to_dict('records')
        res.load(data)

        return res

    # Dump data
    def to_csv(self, cols=None, header=True):
        """
        Returns a ';' delimited csv string with curve content.
        :param cols: Columns to add after timestamp. All ones by default
        :param header: Adds column header roe or not. True by default
        :return:
        """
        csvfile = StringIO()
        if cols is not None:
            cols = [self.datetime_field] + cols
        self.curve.to_csv(
            csvfile, sep=';', columns=cols, index=False, date_format='%Y-%m-%d %H:%M:%S%z', header=header
        )
        return csvfile.getvalue()

    def get_complete_daily_subcurve(self):
        """
        Returns partial curve from first hour to last complete day without gaps nor duplicities
        :return: dataframe
        """
        first_gap = None
        iscomplete, gap = self.is_complete()
        if gap is not None:
            first_gap = gap

        hasduplicates, gap = self.has_duplicates()
        if gap is not None:
            if first_gap is None:
                first_gap = gap
            else:
                first_gap = min(first_gap, gap)

        if first_gap is None:
            return self
        else:
            last_hour = TIMEZONE.normalize(first_gap - timedelta(hours=1))
            if last_hour.hour > 0:
                last_hour = last_hour.replace(hour=0)
            if last_hour >= self.start:
                data = self.curve[self.curve[self.datetime_field] <= last_hour]
                data = data.to_dict('records')
                res = self.__class__()
                res.load(data, datetime_field=self.datetime_field)
            else:
                res = self.__class__()
            return res


    def get_n_rows(self, cols, keep, n=1, order='desc'):
        """
        Returns a new Dataframe with given rows
        :param cols: List of columns
        :param n: Number of rows wanted
        :param keep: Row to keep
        :param order: If you want min or max value
        'first' (asc), 'last' (desc), 'all' all rows
        :return: Dataframe
        """
        if order not in ['asc', 'desc']:
            raise ValueError("ERROR: [order] is not a valid parameter, given keep: {}."
                             "Valid keep options are 'asc', 'desc'".format(order))

        if keep not in ['first', 'last', 'all']:
            raise ValueError("ERROR: [keep] is not a valid parameter, given keep: {}."
                             "Valid keep options are 'first', 'last, 'all'".format(keep))

        if not isinstance(cols, list):
            raise TypeError("ERROR: [cols] has to be a list, given keep: {}.".format(cols))

        if order == 'asc':
            return self.curve.nsmallest(n, cols, keep)
        return self.curve.nlargest(n, cols, keep)

    def _check_magn_is_valid(self, magn):
        """
        Returns True or ValueError if the magn is not valid
        :param magn: Magnitude
        :return: bool
        """
        if magn in DEFAULT_DATA_FIELDS_NO_FACT:
            return True
        raise ValueError("ERROR: [magn] is not a valid parameter, given magn: {}".format(magn))

    @staticmethod
    def convert_numpydate_to_datetime(date, to_string=False):
        import numpy
        str_date = numpy.datetime_as_string(date, unit='s').replace('T', ' ')

        if to_string:
            return str_date

        return datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

    def fill_gaps(self, datetime_from, datetime_to, default_data=None):
        """
        Fills the gaps in curve between **date_from** and **date_to** assigning default values specified in
        **default_data**
        :param datetime_from: localized start datetime
        :param datetime_to: localized end datetime
        :param default_data: dict with field and default value, ie: {'ai': 0, 'ae': 0, 'cch_bruta': False}
        """
        if default_data is None:
            default_data = {'ai': 0.0, 'ae': 0.0, 'r1': 0.0, 'r2': 0.0, 'r3': 0.0, 'r4': 0.0, 'valid': True, 'cch_fact': False}

        if self.has_duplicates():
            self.curve.drop_duplicates(subset=self.datetime_field)

        # creem un nou dataFrame amb una corba segons valors 'default_data'
        pp_fill = PowerProfile(self.datetime_field)
        pp_fill.fill(default_data, datetime_from, datetime_to)

        # combinem de les dues curves per omplir els forats
        self.curve = self.curve.combine_first(pp_fill.curve)

    def apply_chauvenet(self, magn='ai'):
        new_pp = self.copy()

        # Calcular la media
        avg = new_pp.avg(magn)

        # Calcular la desviación estándar
        std = new_pp.std(magn)

        # Número de datos
        leng = len(new_pp.curve[magn])

        # Calcular el límite de desviación estándar usando el criterio de Chauvenet
        criterion = 1 / (2 * leng)

        # Calcular Z_max para el criterio de Chauvenet
        Z_max = math.sqrt(2) * math.erfc(criterion)

        # Calcular Z-scores para los datos
        Z_score = abs(new_pp.curve[magn] - avg) / std

        # Afegim dades al PowerProfile
        new_pp.Z_max = Z_max
        new_pp.std = std
        new_pp.avg = avg
        new_pp.criterion = criterion
        new_pp.curve['Z_score'] = abs(new_pp.curve[magn] - avg) / std
        new_pp.curve['outliers'] = Z_score > Z_max

        # Identificar los valores atípicos según el criterio de Chauvenet
        new_pp.curve = new_pp.curve[Z_score < Z_max]

        return new_pp


class PowerProfileQh(PowerProfile):

    SAMPLING_INTERVAL = 900

    def has_duplicates(self):
        return self.has_duplicates_counter(self.quart_hours)

    def is_complete(self):
        return self.is_complete_counter(self.quart_hours)

    def get_hourly_profile(self):
        """
        Returns a Powerprofile aggregating quarter-hour curve by hour
        :return:
        New Powerprofile
        """

        new_curve = PowerProfile()

        new_curve.curve = self.curve.resample('1H', closed='right', label='right', on=self.datetime_field).sum()
        new_curve.curve.sort_values(by=new_curve.datetime_field, inplace=True)
        new_curve.curve = new_curve.curve.reset_index()
        new_curve.start = new_curve.curve[new_curve.datetime_field].min()
        new_curve.end = new_curve.curve[new_curve.datetime_field].max()

        return new_curve
