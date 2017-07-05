"""Utiltiy functions for workign on the NinaPro Databases (1 & 2)."""

from setuptools import setup, find_packages


setup(name='nina_helper',
      version='2.2',
      description='Utiltiy functions for workign on the NinaPro Databases (1 & 2)',
      author='Lif3line',
      author_email='adamhartwell2@gmail.com',
      license='MIT',
      packages=find_packages(),
      url='https://github.com/Lif3line/nina_helper_package_mk2',  # use the URL to the github repo
      # download_url='https://github.com/Lif3line/nina_helper_package_mk2/archive/2.2.tar.gz',  # Hack github address
      install_requires=[
          'scipy',
          'sklearn',
          'numpy'
      ],
      keywords='ninapro emg')
