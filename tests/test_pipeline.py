"""No model downloads: exercise files, CLIs, fusion, metrics and application."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.ablate import main as ablate
from src.demo import main as demo
from src.evaluate import evaluate_run
from src.io import RunWriter,write_json
from src.prepare_data import validate_manifest


class PipelineTests(unittest.TestCase):
    def test_frozen_run_to_ablations_and_application(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root/'baseline'
            meta = dict(model='synthetic_only',depth_type='metric',unit='meter',representation='depth',
                        fps=10.,visualization_range=[0.,5.])
            writer = RunWriter(run,meta)
            samples = []
            for i in range(4):
                depth = np.full((48,64),2.+i*.02,np.float32)
                rgb = np.full((48,64,3),110,np.uint8)
                rgb[12:28,16:40] = 160
                writer.add(depth,rgb,'synthetic',i,dict(sample_id=f'frame{i}',focal_px=60.))
                np.save(root/f'gt{i}.npy',np.full_like(depth,2))
                samples.append(dict(id=f'frame{i}',rgb=f'baseline/rgb/{i:06d}.png',depth=f'gt{i}.npy'))
            writer.finish({'frames':4})
            baseline_hash = (run/'manifest.json').read_bytes()
            manifest = root/'manifest.json'
            write_json(manifest,dict(protocol=dict(min_depth=.1,max_depth=10.),samples=samples))
            self.assertEqual(validate_manifest(manifest)['count'],4)
            scores = evaluate_run(run,manifest)
            self.assertAlmostEqual(scores[0]['AbsRel'],0.)
            ablate(['--run',str(run),'--output',str(root/'ablations'),'--static-roi','0','10','0','10','--manifest',str(manifest)])
            summary = json.loads((root/'ablations/ablation_summary.json').read_text())
            self.assertEqual(len(summary['results']),7)
            self.assertEqual((run/'manifest.json').read_bytes(),baseline_hash)
            # The complete application renders RGB / depth / metric measurement.
            demo(['--run',str(root/'ablations/A6'),'--output',str(root/'demo')])
            self.assertTrue((root/'demo/point_cloud.ply').stat().st_size > 100)
            self.assertTrue((root/'demo/rgb_depth_application.png').is_file())
            measurement = json.loads((root/'demo/measurements.json').read_text())
            self.assertTrue(1.9 < measurement['measurements'][0]['value_m'] < 2.1)
            self.assertTrue((root/'ablations/A6/visualization/rgb_depth.mp4').stat().st_size>100)

    def test_changed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root/'baseline'
            writer = RunWriter(run,dict(model='synthetic',depth_type='metric',unit='meter',representation='depth'))
            writer.add(np.ones((4,4)),np.zeros((4,4,3),np.uint8),'synthetic',0,dict(sample_id='one'))
            writer.finish({'frames':1})
            np.save(root/'gt.npy',np.ones((4,4)))
            write_json(root/'manifest.json',dict(protocol=dict(min_depth=.1,max_depth=10.),
                samples=[dict(id='one',rgb='baseline/rgb/000000.png',depth='gt.npy')]))
            np.save(run/'depth/000000.npy',np.full((4,4),2,np.float32))
            with self.assertRaisesRegex(ValueError,'hash changed'):
                evaluate_run(run,root/'manifest.json')


if __name__ == '__main__':
    unittest.main()
