import os

from setuptools import setup

setup(
    name='glint-mask-tools',
    version=os.environ['VERSION'],
    packages=['glint_mask_tools'],
    package_dir={'glint_mask_tools': 'core'},
    install_requires=['numpy', 'scipy', 'pillow'],
    url='https://github.com/HakaiInstitute/glint-mask-tools',
    license='MIT',
    author='Taylor Denouden',
    author_email='taylor.denouden@hakai.org',
    description='Create masks for specular reflection in UAV and aerial imagery.'
)
