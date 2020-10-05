from distutils.core import setup

setup(name='pytg',
      version='0.1.0',
      description='Numerical solver for the viscous Taylor-Goldstein equation.',
#      long_description=open('README.rst').read(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Oceanography :: Data analysis',
      ],
      url='http://github.com/jessecusack/pytg',
      author='Jesse Cusack',
      author_email='jmcusack@marine.rutgers.edu',
      license='MIT',
      packages=['pytg'],
      install_requires=[
          'numpy', 'scipy', 'findiff',
      ],
      python_requires='>=3.5',
      zip_safe=False)
