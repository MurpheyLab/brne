from setuptools import setup
from glob import glob

package_name = 'brne_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Katie Hughes',
    maintainer_email='khughes1@u.northwestern.edu',
    description='A python implementation of the BRNE algorithm for crowd navigation',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brne_nav = brne_py.brne_nav:main',
        ]
    },
)
