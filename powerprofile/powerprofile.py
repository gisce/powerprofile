# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime
from pytz import timezone
import pandas as pd
import copy
from .exceptions import *
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

    def dump(self):

        data = self.curve.to_dict(orient='records')

        return data

    @property
    def hours(self):
        return self.curve.count()[self.datetime_field]

    def is_complete(self):
        ''' Checks completeness of curve '''
        start = self.start
        if self.start.tzinfo is None or self.start.tzinfo.utcoffset(self.start) is None:
            start = TIMEZONE.localize(self.start)
        end = self.end
        if self.end.tzinfo is None or self.end.tzinfo.utcoffset(self.end) is None:
            end = TIMEZONE.localize(self.end)

        hours = ((end - start).total_seconds() + 3600) / 3600
        if self.hours != hours:
            return False
        return True

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

    def has_duplicates(self):
        ''' Checks for duplicated hours'''
        uniques = len(self.curve[self.datetime_field].unique())
        if uniques != self.hours:
            return True
        return False

    def check(self):
        '''Tests curve validity'''
        if self.has_duplicates():
            raise PowerProfileDuplicatedTimes
        if not self.is_complete():
            raise PowerProfileIncompleteCurve
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
            self.curve.iloc[item.start]
            self.curve.iloc[item.stop]
            powpro = PowerProfile()
            powpro.curve = res
            powpro.start = res.iloc[0][self.datetime_field]
            powpro.end = res.iloc[-1][self.datetime_field]
            return powpro
        elif isinstance(item, datetime):
            if not datetime.tzinfo:
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
        totals = self.curve.sum()
        res = {}
        for magn in magns:
            res[magn] = totals[magn]
        return res

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

    # Unary
    def copy(self):
        """
        Returns an identical copy of the same profile
        :return: PowerProfile Object
        """
        new = PowerProfile(self.datetime_field)
        new.start = self.start
        new.end = self.end
        new.curve = copy.copy(self.curve)

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

        return new

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


class PowerProfileQh(PowerProfile):

    @property
    def hours(self):
        return self.curve.count()[self.datetime_field] / 4.0

    @property
    def quart_hours(self):
        return self.curve.count()[self.datetime_field]

    def has_duplicates(self):
        ''' Checks for duplicated hours'''
        uniques = len(self.curve[self.datetime_field].unique())
        if uniques != self.quart_hours:
            return True
        return False

    def is_complete(self):
        ''' Checks completeness of curve '''
        start = self.start
        if self.start.tzinfo is None or self.start.tzinfo.utcoffset(self.start) is None:
            start = TIMEZONE.localize(self.start)
        end = self.end
        if self.end.tzinfo is None or self.end.tzinfo.utcoffset(self.end) is None:
            end = TIMEZONE.localize(self.end)

        quart_hours = (((end - start)).total_seconds() + 900) / 900
        if self.quart_hours != quart_hours:
            return False
        return True
