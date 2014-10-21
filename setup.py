from setuptools import setup, find_packages
import codecs
import optoy

def long_description():
    with codecs.open('README.rst', encoding='utf8') as f:
        return f.read()

setup(
        name='optoy',
        version=optoy.__version__,
        description=optoy.__doc__.strip(),
        long_description=long_description(),
        url='http://optoy.casadi.org/',
        download_url='https://github.com/casadi/optoy',
        author=optoy.__author__,
        author_email='joris@casadi.org',
        license=optoy.__license__,
        packages=find_packages(),
        )
