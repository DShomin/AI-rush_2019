#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2

from distutils.core import setup
setup(
    name='airush1',
    version='1.0',
    install_requires=['lightgbm>=2.2.3', 'xgboost>=0.90', 'catboost>=0.16.5']

)

