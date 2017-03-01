from setuptools import setup

setup(name='tflab',
      version='0.1',
      description='A laboratory for experimenting with Tensorflow sbstraction',
      url='https://github.com/mhamilton723/tflab',
      author='Mark Hamilton',
      author_email='mhamilton723@gmail.com',
      license=None,
      packages=['tflab'],
      zip_safe=False,
      install_requires=[
          'tensorflow',
          'numpy']
      )
