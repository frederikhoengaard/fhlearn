from setuptools import setup, find_packages

with open('requirements.txt') as infile:
    requirements = infile.readlines()

setup(
    name='fhlearn',
    version = '0.2.0',
    url = 'https://github.com/frederikhoengaard/fhlearn/tree/main',
    download_url = 'https://github.com/frederikhoengaard/fhlearn/tree/main',
    license = '',
    author = 'Frederik P. Høngaard',
    author_email = 'mail@frederikhoengaard.com',
    description = 'Machine learning model library',
    packages = find_packages(),
    install_requires = requirements
)