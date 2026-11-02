import cv2
import numpy as np
import pytest
from src.package_submission import package_course
from src.io import RunWriter,read_frames,sha256
from src.exploration.edge_aware import CUDAEdgeRefiner,refine_edges


def fixture_video(tmp_path):
    file=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'mp4v'),24,(32,24))
    if not writer.isOpened():pytest.skip('MP4 codec unavailable')
    for value in (20,100):writer.write(np.full((24,32,3),value,np.uint8))
    writer.release()
    return file


def test_short_asset_cannot_silently_satisfy_requested_count(tmp_path):
    file=fixture_video(tmp_path)
    manifest=dict(source_video=file.name,requested_frames=100,runs=[
        dict(name='dav2',role='framewise_relative',path='a'),dict(name='vda',role='video_relative',path='b')])
    with pytest.raises(ValueError,match='acknowledgement'):
        package_course(manifest,tmp_path,tmp_path/'output.zip')


def test_duplicate_source_indices_are_not_a_real_video_prefix(tmp_path):
    file=fixture_video(tmp_path);frames,_,fps=read_frames(file,max_side=640)
    for name,indices in [('a',[0,1]),('b',[0,0])]:
        writer=RunWriter(tmp_path/name,dict(depth_type='relative',unit='arbitrary',representation='inverse_depth',alignment='none',
            frame_indices=indices,fps=fps,source_sha256=sha256(file)))
        for i,rgb in enumerate(frames):writer.add(np.ones((24,32),np.float32),rgb,file,i)
        writer.finish(dict(frames=2))
    manifest=dict(source_video=file.name,requested_frames=100,short_source_acknowledged=True,runs=[
        dict(name='dav2',role='framewise_relative',path='a'),dict(name='vda',role='video_relative',path='b')])
    with pytest.raises(ValueError,match='consecutive frames'):
        package_course(manifest,tmp_path,tmp_path/'output.zip')


def test_relative_support_includes_zero_endpoint(tmp_path):
    writer=RunWriter(tmp_path/'relative',dict(depth_type='relative',unit='arbitrary',representation='inverse_depth'))
    writer.add(np.array([[0,1],[2,0]],np.float32),np.zeros((2,2,3),np.uint8),'unit_test')
    mask=cv2.imread(str(tmp_path/'relative/valid/000000.png'),0)
    assert (mask==255).all()


def test_RGB_guidance_protects_real_depth_step_and_does_not_copy_texture():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    z=np.full((24,32),2,np.float32);z[:,16:]=4
    rgb=np.zeros((24,32,3),np.uint8);rgb[:,16:]=255
    guided=CUDAEdgeRefiner(use_depth_guidance=False)(z,rgb)
    plain=CUDAEdgeRefiner(use_rgb_guidance=False,use_depth_guidance=False)(z,rgb)
    assert np.mean(abs(guided-z))<np.mean(abs(plain-z))*.01
    constant=np.full_like(z,2)
    np.testing.assert_allclose(CUDAEdgeRefiner()(constant,rgb),constant,rtol=1e-6)
    np.testing.assert_allclose(CUDAEdgeRefiner()(z,rgb),refine_edges(z,rgb),rtol=2e-5)
