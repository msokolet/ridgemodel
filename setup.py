#!/usr/bin/env python
# Install script for widefield tools

#  wfield - tools to analyse widefield data
# Copyright (C) 2020 Joao Couto - jpcouto@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
from setuptools.command.install import install
import os
from os.path import join as pjoin

ridgemodel_dir = pjoin(os.path.expanduser('~'),'.ridgemodel')
if not os.path.isdir(ridgemodel_dir):
    print('Creating {0}'.format(ridgemodel_dir))
    os.makedirs(ridgemodel_dir)

longdescription = '''Create design matrix and perform ridge regression on widefield imaging data. Based on a MATLAB package by Simon Musall: https://github.com/churchlandlab/ridgeModel'''
setup(
    name = 'ridgemodel',
    version = '0.1',
    author = 'Michael Sokoletsky',
    author_email = 'michael.sokoletsky@weizmann.ac.il',
    url = 'https://github.com/msokolet/ridgemodel',
    description = (longdescription),
    long_description = longdescription,
    license = 'MIT',
    packages = ['ridgemodel'],
    entry_points = {
        'console_scripts': [
            'wfield = wfield.cli:main'
        ]
    }
)
