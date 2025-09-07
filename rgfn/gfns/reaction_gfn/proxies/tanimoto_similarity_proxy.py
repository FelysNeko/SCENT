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
    def __init__(self, smiles: str, similarity_is_1_penalty: float):
        super().__init__()
        mol = Chem.MolFromSmiles(smiles)
        assert mol
        assert 0.0 <= similarity_is_1_penalty <= 1.0
        self.reference_smiles = smiles
        self.similarity_is_1_penalty = similarity_is_1_penalty
        self.reference_ecfp6 = AllChem.GetMorganFingerprint(mol, 3)
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_tanimoto_similarity(self, smiles: str) -> float:
        if self.reference_smiles == smiles:
            return 1.0 - self.similarity_is_1_penalty
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0
        ecfp6 = AllChem.GetMorganFingerprint(mol, 3)
        return Chem.DataStructs.TanimotoSimilarity(self.reference_ecfp6, ecfp6)

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        return [self._compute_tanimoto_similarity(s.molecule.smiles) for s in states]
