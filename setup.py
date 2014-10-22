from setuptools import setup, find_packages
import codecs

with open('optoy/conf.py') as f:
  exec(f.read())

def long_description():
    with codecs.open('README.rst', encoding='utf8') as f:
        return f.read()

setup(
        name='optoy',
        version=__version__,
        description=__description__,
        long_description=long_description(),
        url='http://optoy.casadi.org/',
        download_url='https://github.com/casadi/optoy',
        author=__author__,
        author_email='joris@casadi.org',
        license=__license__,
        packages=find_packages(),
        )
