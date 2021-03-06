from setuptools import setup, find_packages

setup(
    name='synthpop',
    version='0.1.1',
    description='Population Synthesis',
    author='UrbanSim Inc.',
    author_email='udst@urbansim.com',
    license='BSD',
    url='https://github.com/udst/synthpop',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.8'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'census>=0.5',
        'numexpr>=2.3.1',
        'numpy>=1.8.0',
        'pandas>=0.15.0',
        'scipy>=0.13.3',
        'us>=0.8'
    ]
)
