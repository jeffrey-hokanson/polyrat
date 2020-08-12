import os
from setuptools import setup


install_requires = [
	'numpy',
	'scipy',
	'iterprinter',
	] 

# CVXPY is licensed under the GPL
# My understanding is that this implies the _tests_ are under a GPL license
# but the underlying code in polyrat is still able to be under a more permissive license
# See discussion: https://opensource.stackexchange.com/a/7510
test_requires = [
	'pytest',
	'cvxpy',
]

with open('README.md', 'r') as f:
	long_description = f.read()

setup(name='polyrat',
	version = '0.1',
	url = 'https://github.com/jeffrey-hokanson/PolyRat',
	description = 'Polynomial and rational function library',
	long_description = long_description,
	long_description_content_type = 'text/markdown', 
	author = 'Jeffrey M. Hokanson',
	author_email = 'jeffrey@hokanson.us',
	packages = ['polyrat',],
	install_requires = install_requires,
	test_requires = test_requires,
	python_requires='>=3.6',
	)
