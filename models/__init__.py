"""
Architectural definitions for raw DNA sequence classification models.
"""

from .genexplorer import GenExplorer
from .fulle import FullE10Ensemble, FullEBranch, FullEMerger
from .nucleic_transformer import NucleicTransformer
from .model_3dcv import Model3DCV
from .helix_cnn import HelixCNN
from .capsnet_2d import CapsEnhancerModel


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
        "Model3DCV",
        "HelixCNN",
        "CapsEnhancerModel",
    ]
except ImportError:
    __all__ = [
        "GenExplorer",
        "FullE10Ensemble",
        "FullEEnsemble",
        "FullEBranch",
        "FullEMerger",
        "NucleicTransformer",
        "Model3DCV",
        "HelixCNN",
        "CapsEnhancerModel",
    ]