import os
import imageio
import matplotlib.pyplot as plt


if __name__ == '__main__':

    root = 'F:/Kanaz/Dementia'
    data_path = os.path.join(root, 'Data/AIMIN_Challenge_Training_Dataset')

#    sample_name = '98050022/980500221101-Brain MRI -C/'
    sample_name = '98050025/980500251101-Brain MRI -C'
    save_path = os.path.join(root, 'Outputs/Samples/', sample_name)
    os.makedirs(save_path, exist_ok=True)
    
    sample_data = os.path.join(data_path, sample_name)

    dirs = os.listdir(sample_data)
    for scan_dir in dirs:
        scn_path = os.path.join(sample_data, scan_dir)
        scn = imageio.volread(scn_path, 'DICOM')
        print(scn.shape)

        nrows = 3
        ncols = 6
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
        for i in range(nrows):
            for j in range(ncols):
                idx = int(i * ncols + j)
                if idx < scn.shape[0]:
                    img = scn[idx, :, :]
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title('%d' % idx)
            
        file_name = os.path.join(save_path, '%s.png' % scan_dir)
        fig.savefig(file_name)
#    plt.show()

