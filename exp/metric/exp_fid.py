from pathlib import Path
from ACGPN.test import run_test as run_acgpn
from C_VTON.test import foo as run_cvton
from FS_VTON.test import run_test as run_fs_vton
from PF_AFN.test import run_test as run_pfafn
from RMGN_VITON.test import run_test as run_rmgn
from DAFlow.test import run_test as run_daflow
from metrics.pytorch_fid.fid_score import calculate_fid_given_paths
from metrics.lpips.lpips import calculate_lpips_given_paths


DEVICE = 'cuda:1'
MODELS = {
    'ACGPN': run_acgpn,
    'C_VTON': run_cvton, 
    'DAFlow': run_daflow,
    # 'RMGN_VITON': run_rmgn,
    'PF_AFN': run_pfafn,
    'FS_VTON': run_fs_vton,
}

for model_path, run_func in MODELS.items():
    run_func(Path(model_path) / 'VITON/VITON_test', Path('results') / model_path, batch_size=1, device=DEVICE)

    test_dir = Path(model_path) / 'VITON/VITON_test' / 'test_img'
    out_dir = Path('results') / model_path / 'tryon'
    fid = calculate_fid_given_paths(paths=['VITON/VITON_test/test_img', str(out_dir)], batch_size=50, device=DEVICE)
    lpips = calculate_lpips_given_paths(paths=['VITON/VITON_test/test_img', str(out_dir)], device=DEVICE)
    print(f'{model_path}: {fid}, {lpips}')