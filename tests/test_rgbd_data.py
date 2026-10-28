import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import cv2
import numpy as np

from src.data.tum import associate,interpolate_pose,prepare_sequence
from src.data.http_range import HTTPRangeFile
from src.metrics import evaluation_mask
from src.metrics.pose import pose_temporal_metrics,pose_matrix


class RGBDDataTests(unittest.TestCase):
    def test_association_is_one_to_one_and_bounded(self):
        rgb=[(1.,['a']),(1.01,['b']),(2.,['c'])]
        depth=[(1.002,['d']),(2.1,['e'])]
        pairs=associate(rgb,depth,.02)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0][0][1],['a'])

    def test_pose_interpolation_and_gap_rejection(self):
        poses=[(0.,['0','0','0','0','0','0','1']),(.04,['2','0','0','0','0','1','0'])]
        pose=interpolate_pose(poses,.02)
        self.assertAlmostEqual(pose[0],1.)
        self.assertAlmostEqual(pose[-1],2**-.5)
        self.assertIsNone(interpolate_pose(poses,-.01))
        self.assertIsNone(interpolate_pose(poses,.02,max_gap=.01))

    def test_pose_compensation_removes_camera_translation(self):
        previous=np.full((12,16),2.,np.float32)
        current=np.full_like(previous,1.8)
        p0=[0,0,0,0,0,0,1]
        p1=[0,0,.2,0,0,0,1]
        result=pose_temporal_metrics(previous,current,previous,current,p0,p1,dict(fx=20.,fy=20.,cx=7.5,cy=5.5))
        self.assertLess(result['pose_temporal_MAE_m'],1e-6)
        self.assertGreater(result['coverage'],.5)

    def test_pose_occlusion_mask_excludes_inconsistent_surface(self):
        old=np.full((4,5),2.,np.float32)
        new=np.full_like(old,1.)
        pose=[0,0,0,0,0,0,1]
        result=pose_temporal_metrics(old,new,old,new,pose,pose,dict(fx=10.,fy=10.,cx=2.,cy=1.5))
        self.assertIsNone(result['pose_temporal_log_error'])

    def test_exclusive_bounds_preserve_frozen_nyu_protocol(self):
        gt=np.array([[.001,1.,10.]])
        np.testing.assert_array_equal(evaluation_mask(gt,.001,10.,bounds='exclusive'),[[False,True,False]])

    def test_cached_range_reader_seek_and_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            remote=HTTPRangeFile('https://example.invalid/sample',11,directory,block_size=4)
            content=b'hello world'
            for start in range(0,11,4):
                end=min(10,start+3)
                file=remote.cache/f'{start}-{end}.bin'
                file.write_bytes(content[start:end+1])
                file.with_suffix('.headers').write_text(f'HTTP/1.1 206\nContent-Range: bytes {start}-{end}/11\n')
            remote.seek(3)
            self.assertEqual(remote.read(5),b'lo wo')
            remote.seek(-2,2)
            self.assertEqual(remote.read(),b'ld')
            self.assertEqual(remote.read(),b'')
            remote.close()

    def test_tum_import_preserves_registered_png_and_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            archive=root/'sample.tgz'
            prefix='rgbd_dataset_freiburg1_xyz/'
            rgb=np.full((8,10,3),100,np.uint8)
            depth=np.full((8,10),10000,np.uint16)
            depth[0,0]=0
            files={'rgb.txt':'1.0 rgb/one.png\n1.1 rgb/two.png\n','depth.txt':'1.002 depth/one.png\n1.102 depth/two.png\n',
                   'groundtruth.txt':'\n'.join(f'{t:.2f} 0 0 0 0 0 0 1' for t in np.arange(.98,1.141,.02))}
            with tarfile.open(archive,'w:gz') as tar:
                for filename,value in list(files.items()):
                    data=value.encode()
                    member=tarfile.TarInfo(prefix+filename); member.size=len(data)
                    tar.addfile(member,io.BytesIO(data))
                for kind,image in [('rgb',rgb),('depth',depth)]:
                    ok,buffer=cv2.imencode('.png',image); self.assertTrue(ok)
                    for name in ('one','two'):
                        member=tarfile.TarInfo(prefix+kind+'/'+name+'.png'); member.size=len(buffer)
                        tar.addfile(member,io.BytesIO(buffer.tobytes()))
            manifest=prepare_sequence(archive,root/'prepared',max_frames=2)
            self.assertEqual(len(manifest['samples']),2)
            self.assertEqual(manifest['samples'][0]['depth_scale'],.0002)
            np.testing.assert_array_equal(pose_matrix(manifest['samples'][0]['pose_world_camera']),np.eye(4))
            self.assertTrue((root/'prepared/validation.json').exists())


if __name__=='__main__': unittest.main()
