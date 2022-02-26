from setuptools import setup, find_packages

setup(
    name                = 'Easy Labeling',
    version             = '0.1',
    description         = 'this package makes your labeling easier',
    author              = 'Lee-Ji-won104',
    author_email        = 'bungae104@gmail.com',
    url                 = 'https://github.com/Lee-Ji-won104/Easy-Labeling',
    download_url        = 'https://github.com/Lee-Ji-won104/Easy-Labeling/archive/0.0.tar.gz',
    install_requires    =  [],
    packages            = find_packages(exclude = []),
    keywords            = ['easylabeling'],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)