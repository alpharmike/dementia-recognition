import pydicom as dicom
import imageio
import matplotlib.pyplot as plt

scn = imageio.volread('AIMIN Challenge - Training Dataset', 'DICOM')
# print(scn.shape)
nrows = 2
ncols = 4
patient_id = str('AIMIN Challenge - Training Dataset'.split('/')[1])
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
for i in range(nrows):
    for j in range(ncols):
        idx = int(i * ncols + j)
        if idx < scn.shape[0]:
            img = scn[idx, :, :]
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_title('%d' % idx)
save_path = 'AIMIN Challenge - Training Dataset' + 'with' + '.png'
fig.savefig(save_path)