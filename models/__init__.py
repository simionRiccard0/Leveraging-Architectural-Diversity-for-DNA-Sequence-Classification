"""
Architectural definitions for raw DNA sequence classification models.
"""

from .genexplorer import GenExplorer
from .fulle import FullE10Ensemble, FullEBranch, FullEMerger
from .nucleic_transformer import NucleicTransformer
from .sequencecnn import SequenceCNN


FullEEnsemble = FullE10Ensemble

try:
    from .dnabert2_camp import DNABERT2CAMP
    __all__ = [
        "GenExplorer",
        "FullE10Ensemble",
        "FullEEnsemble",
        "FullEBranch",
        "FullEMerger",
        "NucleicTransformer",
        "DNABERT2CAMP",
        "SequenceCNN",
    ]
except ImportError:
    __all__ = [
        "GenExplorer",
        "FullE10Ensemble",
        "FullEEnsemble",
        "FullEBranch",
        "FullEMerger",
        "NucleicTransformer",
        "SequenceCNN",
    ]