"""Masked boundary extension must agree with the official dense metric."""
import unittest
import numpy as np
from scripts.run_ibims_refinement_cpu import masked_si
from src.metrics.boundary import _official_si_boundary


class MaskedBoundaryTests(unittest.TestCase):
    def test_dense_case_matches_official(self):
        rng=np.random.default_rng(23)
        pred=rng.uniform(.1,9,(24,32))
        gt=rng.uniform(.1,9,(24,32))
        self.assertAlmostEqual(masked_si(pred,gt,np.ones(gt.shape,bool)),_official_si_boundary()(pred,gt),places=12)

    def test_invalid_values_cannot_create_scored_edges(self):
        gt=np.ones((24,32),np.float32); gt[:,16:]=3
        pred=gt.copy(); valid=np.ones(gt.shape,bool); valid[6:14,6:12]=False
        expected=masked_si(pred,gt,valid)
        gt[~valid]=np.nan; pred[~valid]=5000
        self.assertAlmostEqual(masked_si(pred,gt,valid),expected,places=12)


if __name__=='__main__': unittest.main()
