import os
import numpy as np
from PIL import Image

def generate_synthetic_histology_data(target_dir=r'data\lung_subset_demo', n_per_class=20):
    classes = ['lung_n', 'lung_aca', 'lung_scc']
    for cat in classes:
        cat_dir = os.path.join(target_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(n_per_class):
            # Generate synthetic histopathological H&E-stained patch (purple/pink cellular noise)
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[:, :, 0] = np.random.randint(180, 240, (128, 128)) # Pink/red
            arr[:, :, 1] = np.random.randint(100, 180, (128, 128))
            arr[:, :, 2] = np.random.randint(180, 255, (128, 128)) # Purple nuclei
            img = Image.fromarray(arr)
            img.save(os.path.join(cat_dir, f'sample_{i+1}.png'))
            
    print(f"Generated synthetic histopathology dataset at: {target_dir}")

if __name__ == '__main__':
    generate_synthetic_histology_data()
