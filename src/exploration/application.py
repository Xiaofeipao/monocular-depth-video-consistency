"""Direction F: measurable ROI obstacle warning, optionally with hysteresis."""
import numpy as np


class ObstacleWarning:
    def __init__(self, threshold_m=1.5, near_fraction=.05, exit_margin_m=.15, confirmations=2):
        if threshold_m<=0 or not 0<near_fraction<=1 or exit_margin_m<0 or confirmations<1:
            raise ValueError('Invalid warning settings')
        self.threshold,self.fraction,self.margin = threshold_m,near_fraction,exit_margin_m
        self.confirmations = confirmations
        self.state,self.pending = False,0

    def step(self, depth, roi=None):
        values = depth if roi is None else depth[roi[0]:roi[1],roi[2]:roi[3]]
        valid = np.isfinite(values)&(values>0)
        if not valid.any():
            # Missing data must never be represented as a safe measurement.
            return {'warning':None,'near_distance_m':None,'coverage':0.}
        distance = float(np.quantile(values[valid],self.fraction))
        threshold = self.threshold+self.margin if self.state else self.threshold
        desired = distance < threshold
        self.pending = self.pending+1 if desired!=self.state else 0
        if self.pending>=self.confirmations:
            self.state,self.pending = desired,0
        return {'warning':self.state,'near_distance_m':distance,'coverage':float(valid.mean())}


def warning_metrics(predicted, target):
    if len(predicted)!=len(target) or not target:
        raise ValueError('Warning labels must be paired and nonempty')
    if any(t is None for t in target):
        raise ValueError('Ground-truth warning unavailable')
    # Unavailable prediction counts as incorrect, not silently skipped.
    tp=sum(p is True and t is True for p,t in zip(predicted,target))
    fp=sum(p is True and t is False for p,t in zip(predicted,target))
    fn=sum(p is not True and t is True for p,t in zip(predicted,target))
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    return {'warning_precision':precision,'warning_recall':recall,
            'warning_F1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'warning_accuracy':sum(p is not None and p==t for p,t in zip(predicted,target))/len(target),
            'warning_unavailable':sum(p is None for p in predicted)}
