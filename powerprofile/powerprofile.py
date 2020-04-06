# -*- coding: utf-8 -*-
from datetime import datetime
from pytz import timezone
import pandas as pd
from exceptions import *

TIMEZONE = timezone('Europe/Madrid')

class PowerProfile():

    def __init__(self):

        self.start = None
        self.end = None
        self.curve = None

    def load(self, data ,start=None, end=None, datetime_field='timestamp'):
        if not isinstance(data, (list, tuple)):
            raise TypeError("ERROR: [data] must be a list of dicts ordered by timestamp")

        if start and not isinstance(start, datetime):
            raise TypeError("ERROR: [start] must be a localized datetime")

        if end and not isinstance(end, datetime):
            raise TypeError("ERROR: [end] must be a localized datetime")

        if data and not data[0].get(datetime_field, False):
            raise TypeError("ERROR: No timestamp field. Use datetime_field option to set curve datetime field")

        self.curve = pd.DataFrame(data)

        if datetime_field != 'timestamp':
            self.curve = self.curve.rename(columns={datetime_field: 'timestamp'})

        if start:
            self.start = start
        else:
            self.start = self.curve['timestamp'].min()

        if end:
            self.end = end
        else:
            self.end = self.curve['timestamp'].max()

    @property
    def hours(self):
        return self.curve.count()['timestamp']

    def is_complete(self):
        ''' Checks completeness of curve '''
        hours = (((self.end - self.start)).total_seconds() + 3600) / 3600
        if self.hours != hours:
            return False
        return True

    def has_duplicates(self):
        ''' Checks for duplicated hours'''
        uniques = len(self.curve['timestamp'].unique())
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
            powpro.start = res.iloc[0]['timestamp']
            powpro.end = res.iloc[-1]['timestamp']
            return powpro
        elif isinstance(item, datetime):
            if not datetime.tzinfo:
                raise TypeError('Datetime must be a localized datetime')

            res = self.curve.loc[self.curve['timestamp'] == item]
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
