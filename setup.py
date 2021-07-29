from setuptools import setup

setup(
    name='nengo_backends',
    packages=['nengo_lmublock'],
    version='0.0.1',
    description='A collection of nengo backends for different hardware and software',
    url='https://github.com/neuromorphs/ant-nengo-backends',
    license='GPLv3',
    install_requires=['numpy', 'nengo'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ]
    )
