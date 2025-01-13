from setuptools import setup

setup(
   name='PYPGLM',
   version='0.99.10',
   description='This package is designed to facilitate the modeling of probabilistic, graphical, logical systems.',
   author='Salma Bayoumi, De Landtsheer Sebastien',
   author_email='salma.ismail.hamed@gmail.com',
   packages=['PYPGLM'], 
   install_requires=[
    "networkx",
    "numpy",
    "pandas",
    "emoji",
    "pytest",
    "pytest-cov",
    "openpyxl",
    "scipy",
    "matplotlib",
    "gaft",
    "seaborn",
    "joblib",
    "scikit-learn"
    ], long_description=open('README.md').read(),
    long_description_content_type="text/markdown"
)