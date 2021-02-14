from setuptools import setup, find_packages
import os

here = os.path.dirname(__file__)

try:
  with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
except FileNotFoundError:
  long_description = ''

try:
  with open(os.path.join(here, 'VERSION'), encoding='utf-8') as f:
    version = f.read().strip()
except FileNotFoundError:
  version = '0.0.0'

setup(
  # please, change for your project
  name='roboscientist',
  version=version,
  description="""Active learning of symbolic regressions""",

  long_description=long_description,
  long_description_content_type="text/markdown",

  # please, change this for your project
  url='',

  # please, change this for your project
  author='Vladislav Belavin',
  author_email='belavin@phystech.edu',

  # please, change this for your project
  maintainer='Vladislav Belavin',
  maintainer_email='belavin@phystech.edu',

  # you might also want to change the license
  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],

  packages=find_packages(where='src', ),
  package_dir={'': 'src'},

  extras_require={
    'test': [
      'pytest >= 4.0.0',
    ],
  },

  install_requires=[
    'numpy >= 1.18.0',
    'scipy >= 1.4.0',
    'scikit-learn >= 0.20',
    'matplotlib >= 3.1.0',
    'torch >= 1.4.0',
    'torchvision >= 0.5.0',
    'tqdm >= 4.40.2',
    'comet_ml >= 3.1.3', 'networkx', 'sympy'
  ],

  python_requires='>=3.5'
)
