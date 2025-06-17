




# import PIL for loading tif
import numpy as np

import tifffile

from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.morphology import disk

from scipy.ndimage import convolve

import matplotlib.pyplot as plt

cm_to_inch = 1/2.54

# 
outpudir = '/Users/m.wehrens/Data_UVA/2024_small-analyses/KevinPeek/python_out/'
# import two images
img1_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/KevinPeek/example_data_Kevin/20250527 Developmental series KPM79-80 min35S DMR1-2-TR.lif - KPM79-5_V6L6_oracle1.tif'
img2_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/KevinPeek/example_data_Kevin/20250527 Developmental series KPM79-80 min35S DMR1-2-TR.lif - KPM80-1_V5L5_oracle1.tif'
# load the images
img1 = tifffile.imread(img1_path)[0]
img2 = tifffile.imread(img2_path)[0]

img = img1

# first convolute with a disk
disk_mask = disk(2)/np.sum(disk(2))
img_convolved = convolve(img, disk_mask)

# now again find local maxima
peaks, vals = get_top_n_local_maxima(img_convolved, n=30, min_distance=8)

# plot
fig, ax = plt.subplots(1,2, figsize=(15*cm_to_inch, 10*cm_to_inch))
ax[0].imshow(img_convolved)
ax[0].scatter(peaks[:, 1],peaks[:, 0],
            facecolors='none',edgecolors='white',s=100,linewidths=1,marker='s')
ax[1].imshow(img)
ax[1].scatter(peaks[:, 1],peaks[:, 0],
            facecolors='none',edgecolors='white',s=100,linewidths=1,marker='s')
plt.show(); plt.close()


################################################################################
# Extend method using convolution with disk


img1_conv = convolve_with_disk(img1)
img2_conv = convolve_with_disk(img2)

peaks1, vals1 = get_top_n_local_maxima(img1_conv, n=30, min_distance=8)
peaks2, vals2 = get_top_n_local_maxima(img2_conv, n=30, min_distance=8)

# Plot for image
img_convolved = img1_conv; img=img1; peaks = peaks1
img_convolved = img2_conv; img=img2; peaks = peaks2

fig, ax = plt.subplots(1,2, figsize=(15*cm_to_inch, 10*cm_to_inch))
ax[0].imshow(img_convolved)
ax[0].scatter(peaks[:, 1],peaks[:, 0],
            facecolors='none',edgecolors='white',s=100,linewidths=1,marker='s')
ax[1].imshow(img)
ax[1].scatter(peaks[:, 1],peaks[:, 0],
            facecolors='none',edgecolors='white',s=100,linewidths=1,marker='s')
plt.show(); plt.close()

################################################################################

import pandas as pd
df_intensities = pd.concat([pd.DataFrame({'label':'img1', 'vals':vals1}),
                            pd.DataFrame({'label':'img2', 'vals':vals2})])


# now plot the gathered statistics using seaborn
import seaborn as sns
fig, ax = plt.subplots(figsize=(5.2*cm_to_inch, 5.2*cm_to_inch))
plt.rcParams.update({'font.size': 8})
# sns.boxplot(x='label', y='vals', data=df_intensities)
sns.violinplot(x='label', y='vals', data=df_intensities, inner=None, facecolor='lightgrey', linewidth=0.5, ax=ax)
sns.stripplot(x='label', y='vals', data=df_intensities, color='black', ax=ax, size=1)
# rotate x axis labels 90 deg
plt.xticks(rotation=90)
# add the level of saturated pixels as a horizontal line
ax.axhline(np.iinfo(img1.dtype).max, color='red', linestyle='--', label='background level', linewidth=.5)
# add the background level as horizontal line
ax.axhline(background_lvl_max, color='blue', linestyle='--', label='background level img1', linewidth=.5)
ax.set_ylabel('Intensities of brightest')
ax.set_xlabel('')
plt.tight_layout()
# plt.show()
# save the image to outpudir
plt.savefig(outpudir + 'quantify_strategy2_intensities.pdf', dpi=300, bbox_inches='tight')
plt.close()





