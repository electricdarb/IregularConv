from setuptools import setup, find_packages

with open("README.rst", 'r') as f:
    long_description = f.read()
    print(long_description)

setup(
    name='IrregConv',
    version='0.7',
    license='MIT',
    author="Bradford Gill",
    description='easy to use Irregularly shaped convolution kernels',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email='bradfordgill@umass.edu',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='',
    keywords='IrregConv irregular convolution brain damage',
    install_requires=[
          'numpy',
      ],
)