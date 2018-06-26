#
# WARNING: do *NOT* run directly this description. Use instead ./install.sh
# script that will install all the dependencies
#
from distutils.core import setup
from setuptools import find_packages

setup(name='bauta',
      version='0.0.1',
      description='Framework and toolset for image segmentation',
      author='Pau Carré Cardona, Michael Zucchetta',
      author_email='pau.carre@gmail.com, michael.zucchetta@gmail.com',
      url='https://github.com/gilt/bauta',
      packages=find_packages(exclude='test'),
     )
