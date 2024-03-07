import os
interval = 10
num_images = 7
skip = interval
depth = 'DPT-depth'
store_dirs = 'GoPro+skip-{}/'.format(skip)
os.makedirs(store_dirs, exist_ok=True)
file = open(os.path.join(store_dirs, 'test_files.txt'), 'w')

for split in ['train', 'val']:
    for dataset in ['GoPro']:#, 'REDS']:    
        img_root = os.path.join('/media/data/prasan/datasets/', dataset, split)
        depth_root = os.path.join('/media/data/prasan/datasets/', dataset, depth)

        directories = sorted(os.listdir(img_root))
        for directory in directories:
            imgs = sorted(os.listdir(os.path.join(img_root, directory)))
            for i in range((num_images//2)*interval, len(imgs)-(num_images//2)*interval, skip):
                for a in range(num_images//2*-1, num_images//2+1):
                    img_path = os.path.join(dataset, split, directory, imgs[i+a*interval])
                    depth_path = os.path.join(dataset, depth, split, directory, imgs[i+a*interval])
                    file.write('{0};{1}\t'.format(img_path, depth_path))

                file.write('\n')

file.close()