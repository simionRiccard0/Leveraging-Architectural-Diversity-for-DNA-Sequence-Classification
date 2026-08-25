"""
Architectural definitions for raw DNA sequence classification models.
"""

from .genexplorer import GenExplorer
from .fulle import FullEEnsemble, FullEBranch, FullEMerger
from .nucleic_transformer import NucleicTransformer

try:
    from .dnabert2_camp import DNABERT2CAMP
    __all__ = [
        "GenExplorer",
        "FullEEnsemble",
        "FullEBranch",
        "FullEMerger",
        "NucleicTransformer",
        "DNABERT2CAMP",
    ]
except ImportError:
    __all__ = [
        "GenExplorer",
        "FullEEnsemble",
        "FullEBranch",
        "FullEMerger",
        "NucleicTransformer",
    ]