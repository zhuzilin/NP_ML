from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.1.0'

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='np_ml',
    version=__version__,
    description='A tool library of classical machine learning algorithms with only numpy.',
    url='https://github.com/zhuzilin/NP_ML',
    download_url='https://github.com/zhuzilin/NP_ML',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='zhuzilin',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='zhuzilinallen@gmail.com'
)