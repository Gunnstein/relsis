# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='relsis',
      version='0.2.5',
      url='https://github.com/Gunnstein/relsis',
      license='MIT',
      description='Package for reliability and sensitivity analysis, relsis.',
      author='Gunnstein T. Froeseth',
      author_email='gunnstein.t.froseth@ntnu.no',
      packages=find_packages(exclude=["test"]),
      install_requires=[
        'numpy>=1.0']
     )