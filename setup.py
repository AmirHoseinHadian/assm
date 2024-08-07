import os

from assm.version import *
from setuptools import setup, find_packages

PACKAGES = find_packages(exclude=['tests.unit_tests*'])

# Get version and release info, which is all stored in rlssm/version.py
ver_file = os.path.join('assm', 'version.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=REQUIRES,
            include_package_data=True,
            requires=REQUIRES)

if __name__ == '__main__':
    setup(**opts)
