"""Read-only environment check; no automatic dependency installation."""
import importlib.util
import json
import platform
import sys


def main():
    modules = ('numpy','cv2','PIL','torch','torchvision','timm','einops','h5py','scipy')
    installed = {name:importlib.util.find_spec(name) is not None for name in modules}
    result = dict(python=sys.version,executable=sys.executable,platform=platform.platform(),modules=installed)
    if installed['torch']:
        import torch
        result.update(torch=torch.__version__,cuda_available=torch.cuda.is_available())
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
