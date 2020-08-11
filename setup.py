import os
from setuptools import setup


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
	install_requires = [
		'numpy', 
		'scipy', 
		],
	python_requires='>=3.6',
	)
