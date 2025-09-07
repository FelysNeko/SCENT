from typing import List

import gin
from rdkit import Chem
from rdkit.Chem import AllChem

from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class TanimotoSimilarityProxy(CachedProxyBase[ReactionState]):
    def __init__(self, smiles: str):
        super().__init__()
        mol = Chem.MolFromSmiles(smiles)
        assert mol
        self.ecfp6 = AllChem.GetMorganFingerprint(mol, 3)
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_tanimoto_similarity(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0
        ecfp6 = AllChem.GetMorganFingerprint(mol, 3)
        return Chem.DataStructs.TanimotoSimilarity(self.ecfp6, ecfp6)

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        return [self._compute_tanimoto_similarity(s.molecule.smiles) for s in states]
