'''
@author: jimfan
'''
import os
from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


setup(name='Surreal',
      version='0.0.1',
      author='Stanford RL Group',
      author_email='jimfan@cs.stanford.edu',
      url='http://github.com/StanfordRL/Surreal',
      description='Stanford University Repository for Reinforcement Algorithms.',
      # long_description=read('README.rst'),
      keywords=['Reinforcement Learning', 'Deep Learning'],
      license='GPLv3',
      packages=['surreal'],
      entry_points={
        'console_scripts': [
            'a3c = surreal.a3c.train:main'
        ]
      },
      classifiers=[
          "Development Status :: 2 - Pre-Alpha",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Utilities",
          "Environment :: Console",
          "Programming Language :: Python :: 3"
      ],
      # install_requires=read('requirements.txt').splitlines(),
      include_package_data=True,
      zip_safe=False
)
