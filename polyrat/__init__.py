from importlib.metadata import version, PackageNotFoundError

# Bring in the version number
try:
	__version__ = version(__name__)
except PackageNotFoundError:
	# package is not installed
	pass

# Utilities
from .sorted_norm import *

# Polynomial utils
from .index import *
from .basis import *
from .polynomial import *
from .lagrange import *
from .arnoldi import *

# Rational 
from .rational import *
from .vecfit import *
from .aaa import *
from .paaa import *
from .skiter import *
