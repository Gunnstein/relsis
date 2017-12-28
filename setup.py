# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='relsis',
      version='0.0.1',
      url='https://github.com/Gunnstein/relsis',
      license='MIT',
      description='Package for reliability and sensitivity analysis, relsis.',
      author='Gunnstein T. Froeseth',
      author_email='gunnstein.t.froseth@ntnu.no',
      packages=['relsis'],
      install_requires=[
        'numpy>=1.0']
     )