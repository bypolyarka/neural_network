from setuptools import setup
from mlp import __version__

setup(
    name='neural_network',
    version=__version__,
    packages=['mlp', 'tests'],
    url='https://github.com/bypolyarka/neural_ode',
    license='',
    author='solonko',
    author_email='vitaliy.solonko@gmail.com',
    description='University project for the discipline "Neural Networks"'
)
