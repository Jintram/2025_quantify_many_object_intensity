
################################################################################

# Information
# 
# Channel 0: channel of interest, contains nuclei.
# Channel 1: channel with chlorophyll, could be used for normalization (Q: is this autofluorescence?)

################################################################################

# import PIL for loading tif
import numpy as np

import tifffile

from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.morphology import disk

from scipy.ndimage import convolve

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

cm_to_inch = 1/2.54

# 
outpudir = '/Users/m.wehrens/Data_UVA/2024_small-analyses/KevinPeek/python_out/'
# import two images
img1_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/KevinPeek/example_data_Kevin/20250527 Developmental series KPM79-80 min35S DMR1-2-TR.lif - KPM79-5_V6L6_oracle1.tif'
img2_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/KevinPeek/example_data_Kevin/20250527 Developmental series KPM79-80 min35S DMR1-2-TR.lif - KPM80-1_V5L5_oracle1.tif'
# load the images
img1 = tifffile.imread(img1_path)[0]
img2 = tifffile.imread(img2_path)[0]


################################################################################
# Some playing around to test stuff

img = img2

# a nucleus is ±2-4 pixels in diameter
coordinates = peak_local_max(img, min_distance=4)

# get pixel values at those coordinates
pixel_values = img[coordinates[:, 0], coordinates[:, 1]]

# find background levels using the mode
# first apply median filter, with a radius of 2
img_median = median(img, footprint=disk(2))
# get mode from that image, add 1 to make value >0
img_median_mode = np.bincount(img_median.ravel()).argmax()+1

coordinates_selected = coordinates[pixel_values>img_median_mode*3]

# that didn't work so well


# let's try to filter away most of the nuclei
img_median_broad= median(img, footprint=disk(8))
# let's get the 90 percentile of that 
threshold2 = np.percentile(img_median_broad, 99)
print(threshold2)
# still very low value
plt.imshow(img_median_broad, cmap='gray')
plt.show(); plt.close()
plt.hist(img_median_broad.flatten(), bins=100)
plt.show(); plt.close()


# show the image and the maxima
plt.imshow(np.log(img+1), cmap='viridis')
plt.scatter(coordinates_selected[:,1], coordinates_selected[:,0], color='white')
plt.show(); plt.close()


# plot a histogram of the pixel values
plt.hist(pixel_values, bins=100)
plt.axvline(img_median_mode, color='red', linestyle='--', label='mode')
plt.show(); plt.close()

# maybe still have to resort to top X strategy I drew on paper..

# let's get the top 100 max vals from the coordinates
top_n = 100
top_n_indices = np.argsort(pixel_values)[-top_n:]
# get the coordinates of those top n values
coordinates_top_n = coordinates[top_n_indices]

# now plot again
plt.imshow(np.log(img+1), cmap='viridis')
plt.scatter(coordinates_top_n[:,1], coordinates_top_n[:,0], color='white')
plt.show(); plt.close()

# plot a histogram of those top 100
pixel_values_top100 = pixel_values[top_n_indices]
plt.hist(pixel_values_top100, bins=np.arange(-0.5, 256.5, 1))
plt.show(); plt.close()


################################################################################
# now write this in function form to get top 100


def get_top_n_local_maxima(img, n=100, min_distance=4):
    """
    Find the top n local maxima in an image.

    Parameters:
        img (ndarray): Input image.
        n (int): Number of top maxima to return.
        min_distance (int): Minimum distance between maxima.

    Returns:
        coordinates_top_n (ndarray): Array of shape (n, 2) with coordinates of top maxima.
        pixel_values_top_n (ndarray): Array of length n with pixel values at those coordinates.
    """
    
    coordinates = peak_local_max(img, min_distance=min_distance)
    
    pixel_values = img[coordinates[:, 0], coordinates[:, 1]]
    
    if len(pixel_values) < n:
        n = len(pixel_values)
        
    top_n_indices = np.argsort(pixel_values)[-n:]
    
    coordinates_top_n = coordinates[top_n_indices]
    
    pixel_values_top_n = pixel_values[top_n_indices]
    
    return coordinates_top_n, pixel_values_top_n


def estimate_background_level(img):
    """
    Estimate the background level of an image using a broad median filter and the 99th percentile.

    Parameters:
        img (ndarray): Input image.

    Returns:
        background_lvl (float): Estimated background level.
    """
    img_median_broad = median(img, footprint=disk(8))
    background_lvl = np.percentile(img_median_broad, 99)
    return background_lvl


# now apply this to image 1 and image 2
img1_median1px = median(img1, footprint=disk(1))
img2_median1px = median(img2, footprint=disk(1))
coordinates_top_n_img1, top100_vals_img1 = get_top_n_local_maxima(img1_median1px, n=30, min_distance=4)
coordinates_top_n_img2, top100_vals_img2 = get_top_n_local_maxima(img2_median1px, n=30, min_distance=4)

# determine the background levels for the two images
background_lvl_img1 = estimate_background_level(img1)
background_lvl_img2 = estimate_background_level(img2)
# take the max background
background_lvl_max = np.max([background_lvl_img1, background_lvl_img2])

# add both in a dataframe
import pandas as pd
df_intensities = pd.concat([pd.DataFrame({'label':'img1', 'vals':top100_vals_img1}),
                            pd.DataFrame({'label':'img2', 'vals':top100_vals_img2})])


# now plot the 1st image with selected coordinates on top
plt.imshow(np.log(img1+1), cmap='viridis')
plt.scatter(coordinates_top_n_img1[:, 1],coordinates_top_n_img1[:, 0],
            facecolors='none',edgecolors='white',s=100,linewidths=1,marker='s')
plt.title('Top 30 local maxima in img1')
plt.show(); plt.close()


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
plt.savefig(outpudir + 'quantify_strategy1_intensities.pdf', dpi=300, bbox_inches='tight')
plt.close()



################################################################################

def convolve_with_disk(img, radius=2):
    """
    Convolves the input image with a normalized disk mask of given radius.
    The idea here is that nucleu are ±disk shaped, and have the highest signal.
    So by for every pixel calculating the local average in a disk-shaped
    neighborhood, the signal from actual nuclei is amplified, and the maximum
    of the convolution image will lie at the images center.
    
    Parameters:
        img (ndarray): Input image.
        radius (int): Radius of the disk mask.
        
    Returns:
        img_convolved (ndarray): Convolved image.
    """
    disk_mask = disk(radius) / np.sum(disk(radius))
    img_convolved = convolve(img, disk_mask)
    return img_convolved



img1_conv = convolve_with_disk(img1)
img2_conv = convolve_with_disk(img2)

peaks1, vals1 = get_top_n_local_maxima(img1_conv, n=30, min_distance=8)
peaks2, vals2 = get_top_n_local_maxima(img2_conv, n=30, min_distance=8)


# plot for the two images
def plot_peaks_on_images(img_conv, img, peaks, sample_name):
    """
    Plots the convolved image and the original image side by side,
    overlaying the detected peaks.

    Parameters:
        img_conv (ndarray): The convolved image.
        img (ndarray): The original image.
        peaks (ndarray): Array of peak coordinates (N, 2).
    """
    fig, ax = plt.subplots(1, 2, figsize=(15*cm_to_inch, 10*cm_to_inch))
    plt.rcParams.update({'font.size': 8})
    ax[0].set_title('Modified image')
    ax[0].imshow(img_conv)
    ax[0].scatter(peaks[:, 1], peaks[:, 0],
                  facecolors='none', edgecolors='white', s=30, linewidths=.5, marker='s')
    ax[1].set_title('Original image')
    ax[1].imshow(img)
    ax[1].scatter(peaks[:, 1], peaks[:, 0],
                  facecolors='none', edgecolors='white', s=30, linewidths=.5, marker='s')
    plt.tight_layout()
    # plt.show()
    plt.savefig(outpudir + f'quantify_strategy1_convolved_peaks_img{sample_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()   
    
plot_peaks_on_images(img1_conv, img1, peaks1, 'img1')
plot_peaks_on_images(img2_conv, img2, peaks2, 'img2')    


################################################################################
# And plot again

df_intensities_s2 = pd.concat([pd.DataFrame({'label':'img1', 'vals':vals1}),
                            pd.DataFrame({'label':'img2', 'vals':vals2})])

# now plot the gathered statistics using seaborn
fig, ax = plt.subplots(figsize=(5.2*cm_to_inch, 5.2*cm_to_inch))
plt.rcParams.update({'font.size': 8})
# sns.boxplot(x='label', y='vals', data=df_intensities)
sns.violinplot(x='label', y='vals', data=df_intensities_s2, inner=None, facecolor='lightgrey', linewidth=0.5, ax=ax)
sns.stripplot(x='label', y='vals', data=df_intensities_s2, color='black', ax=ax, size=1)
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
plt.savefig(outpudir + 'quantify_strategy1_intensities.pdf', dpi=300, bbox_inches='tight')
plt.close()
