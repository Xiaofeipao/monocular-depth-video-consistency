"""发布只复制白名单，替换只作用于独立干净克隆，不操作远端。"""
from pathlib import Path
import json
import subprocess
import pytest
from scripts.publishing import export_project as exporter
from scripts.publishing import replace_branch as replacer


def git(repository, *args):
    return subprocess.check_output(['git','-C',str(repository),*args],stderr=subprocess.STDOUT).decode().strip()


@pytest.fixture
def publication(tmp_path):
    package=tmp_path/'package'
    result=exporter.export(exporter.ROOT,package)
    repository=tmp_path/'repository';repository.mkdir()
    git(repository,'init','-b','JiongxuChen')
    git(repository,'config','user.name','Local publication test')
    git(repository,'config','user.email','test@example.invalid')
    git(repository,'config','core.hooksPath','/dev/null')
    (repository/'obsolete.txt').write_text('old')
    (repository/'.old-hidden').write_text('old')
    git(repository,'add','-A');git(repository,'commit','-m','Previous files')
    head=git(repository,'rev-parse','HEAD');git(repository,'branch','main')
    return package,repository,head,result


def test_export_covers_all_methods_and_never_resources(publication):
    package,_,_,result=publication
    names=set(json.loads((package/'PROJECT_EXPORT.json').read_text())['files'])
    assert set(exporter.REQUIRED)<=names
    assert not {name.split('/')[0] for name in names}&replacer.BLOCKED
    assert 'docs/stepC_v2.md' in names
    assert not any(name.endswith(('.mp4','.npz','.npy','.pth','.pt')) for name in names)
    assert 'docs/github_upload_part1_part2.md' not in names
    assert Path(result['archive']).is_file()


def test_replacement_deletes_hidden_old_files_but_preserves_history(publication):
    package,repo,head,_=publication
    result=replacer.replace(package,repo,'JiongxuChen',head)
    assert not result['committed'] and not result['pushed']
    assert not (repo/'obsolete.txt').exists() and not (repo/'.old-hidden').exists()
    for name in ('src/part3_c_v2/refine.py','src/refinement/ORIGIN.json','docs/stepC_v2.md'):
        assert (repo/name).read_bytes()==(package/name).read_bytes()
    assert '100755' in git(repo,'ls-files','--stage','scripts/part3_c_v2/direction_c.sh')
    git(repo,'commit','-m','Complete project replacement')
    assert git(repo,'rev-parse','HEAD^')==head
    assert git(repo,'rev-parse','main')==head
    assert not git(repo,'status','--porcelain')


def test_changed_package_is_rejected_before_deletion(publication):
    package,repo,head,_=publication
    (package/'README.md').write_text('modified')
    with pytest.raises(ValueError,match='SHA'):replacer.replace(package,repo,'JiongxuChen',head)
    assert (repo/'obsolete.txt').exists()


def test_untracked_even_ignored_files_cannot_be_deleted(publication):
    package,repo,head,_=publication
    (repo/'.git/info/exclude').write_text('personal.txt\n')
    (repo/'personal.txt').write_text('personal work')
    with pytest.raises(ValueError,match='未跟踪'):replacer.replace(package,repo,'JiongxuChen',head)
    assert (repo/'personal.txt').exists()


def test_wrong_branch_head_and_project_path_are_rejected(publication):
    package,repo,head,_=publication
    for branch, expected in [('other',head),('JiongxuChen','0'*40)]:
        with pytest.raises(ValueError):replacer.replace(package,repo,branch,expected)
        assert (repo/'obsolete.txt').exists()
    with pytest.raises(ValueError,match='独立'):
        replacer.replace(package,exporter.ROOT,'JiongxuChen',head)
