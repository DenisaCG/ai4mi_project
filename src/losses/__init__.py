# Importing a module registers its losses. Add one line per new loss file.
from . import boundary  # noqa: F401
from . import cross_entropy  # noqa: F401
from . import dice  # noqa: F401
from . import diceCE  # noqa: F401
from . import tversky  # noqa: F401
from . import weighted_cross_entropy  # noqa: F401
from . import weightedCEdice
from . import boundary
from . import boundary_dice_ce  # noqa: F401
from . import boundary_schedule  # noqa: F401
from . import hausdorff  # noqa: F401

