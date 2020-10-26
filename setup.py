import os
from setuptools import setup


install_requires = [
	'numpy',
	'scipy',
	'iterprinter',
	'cvxopt',
	'cvxpy',
	] 

try:
	from funtools import cached_property
except ImportError:
	install_requires += ['backports.cached-property']

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

ns = {}
with open('polyrat/version.py') as f:
	exec(f.read(), ns)

version = ns['__version__']

setup(name='polyrat',
	version = version,
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
	classifiers = [
		'Development Status :: 4 - Beta',
		"Programming Language :: Python :: 3",
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',	
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Mathematics'
	]
	)
