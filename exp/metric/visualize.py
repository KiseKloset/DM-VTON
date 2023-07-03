from PIL import Image
from pathlib import Path
import glob
import random


INPUT = '../dataset/Clean-VITON/VITON_test'
MODELS = {
    'ACGPN': 'results/ACGPN/tryon',
    'C_VTON': 'results/C_VTON/tryon', 
    'DAFlow': 'results/DAFlow/tryon',
    # 'RMGN_VITON': 'results/RMGN_VITON/tryon',
    'PF_AFN': 'results/PF_AFN/tryon',
    'FS_VTON': 'results/FS_VTON/tryon',
    'DMVTON': 'results/DMVTON/tryon'
}
SAVE = 'results/visualize'
Path(SAVE).mkdir(parents=True, exist_ok=True)


def get_data_paths(dir: str | list[str], data_formats: list, prefix: str = '') -> list[str]:
    """
    Get list of files in a folder that have a file extension in the data_formats.

    Args:
      dir (str | list[str]): Dir or list of dirs containing data.
      data_formats (list): List of file extensions. Ex: ['jpg', 'png']
      prefix (str): Prefix for logging messages.

    Returns:
      A list of strings.
    """
    try:
        f = []  # data files
        for d in dir if isinstance(dir, list) else [dir]:
            p = Path(d)
            if p.is_dir():
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        data_files = sorted(x for x in f if x.split('.')[-1].lower() in data_formats)
        return data_files
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {dir}: {e}') from e
    

def merge_img(imgs, offset=0, axes='hor'):
    widths, heights = zip(*(i.size for i in imgs))
    if axes=='hor':
        w, h = sum(widths), max(heights)
        new_im = Image.new('RGB', (w, h))
        pos = 0
        for im in imgs:
            new_im.paste(im, (pos, 0))
            pos = pos + im.size[0] + offset
    elif axes=='ver':
        w, h = max(widths), sum(heights)
        new_im = Image.new('RGB', (w, h))
        pos = 0
        for i, im in enumerate(imgs):
            new_im.paste(im, (0, pos))
            pos = pos + im.size[1] + offset  

    return new_im

# Read pairs
with open(Path(INPUT) / 'test_pairs.txt', 'r') as f:
    fs = f.read().strip().split('\n')
    fs = [i.split() for i in fs]

for pair in fs:
    p, c = Image.open(Path(INPUT) / 'test_img' / pair[0]), Image.open(Path(INPUT) / 'test_color' / pair[1])
    inputs = merge_img([p, c], axes='hor')
    # inputs = inputs.resize((p.size[0]//2, p.size[1]))
    
    outputs = []
    for method, p in MODELS.items():
        outputs.append(Image.open(Path(p) / pair[0]))
    outputs = merge_img(outputs, axes='hor')
    
    results = merge_img([inputs, outputs], axes='hor')
    results.save(Path(SAVE) / pair[0])

# special = ['001868_0', '008959_0']
# picks = ['000370_0.jpg', '000750_0.jpg', '001283_0.jpg', '002031_0.jpg', '006155_0.jpg', '006789_0.jpg', '008959_0.jpg', '014612_0.jpg', '015516_0.jpg', '018047_0.jpg', '019243_0.jpg', '019360_0.jpg']
# picks = ['006789_0.jpg', '015516_0.jpg', '018047_0.jpg']
# imgs = []
# for im_name in picks:
#     im_path = Path(SAVE) / im_name
#     im = Image.open(im_path)
#     imgs.append(im)

# final = merge_img(imgs, axes='ver')
# final.save('pick/final/final.jpg')


# a = get_data_paths(SAVE, data_formats=['jpg', 'png'])
# a = [Path(a).name for a in a]
# samples = random.sample(a, k=10)
# print(samples)
samples = ['019454_0.jpg', '011662_0.jpg', '015428_0.jpg', '007397_0.jpg', '001269_0.jpg', '004981_0.jpg', '019243_0.jpg', '006649_0.jpg', '008598_0.jpg', '014785_0.jpg']