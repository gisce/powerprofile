import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [x.strip() for x in f.readlines()]

with open('requirements-dev.txt') as f:
    TEST_REQUIRES = [x.strip() for x in f.readlines()]

with open('README.md') as f:
    readme = f.read()

setup(
    name="powerprofile",
    version="0.23.0",
    author="GISCE-TI, S.L.",
    author_email="devel@gisce.net",
    description=("Library to manage power profiles"),
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/gisce/powerprofile",
    license="GNU Affero General Public License v3",
    install_requires=INSTALL_REQUIRES,
    setup_requires=["pytest-runner"],
    tests_require=TEST_REQUIRES,
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.11",
        "Topic :: Utilities"
    ],
)
