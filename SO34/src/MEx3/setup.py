from setuptools import find_packages, setup

package_name = 'MEx3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='madee',
    maintainer_email='mspangaliman@up.edu.ph',
    description='Image publisher and image subscriber for MEx3',
    license='MIT',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher = MEx3.image_publisher:main',
            'crack_image_publisher = MEx3.crack_image_publisher:main',
            'image_subscriber = MEx3.image_subscriber:main',
            'gazzard_gui = MEx3.gazzard_gui:main',
            'gazzard_gui_sam3 = MEx3.gazzard_gui_sam3:main',
            'gazzard_gui_v2 = MEx3.gazzard_gui_v2:main',
            'fake_co2_publisher = MEx3.fake_co2_publisher:main',
        ],
    },
)
