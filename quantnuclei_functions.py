
#%% ###############################################################################

# import PIL for loading tif
import os

import numpy as np

import tifffile

from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.morphology import disk

from scipy.ndimage import convolve
from scipy.stats import mode

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

cm_to_inch = 1/2.54


#%% ###############################################################################
# Image analysis functions

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
    
    # to visualize the mask:
    # plt.imshow(disk_mask); plt.show(); plt.close()
    
    return img_convolved

#%% ###############################################################################
# Plotting functions for images

# plot for the two images
def plot_peaks_on_images(img_conv, img, peaks, sample_name, outputdir):
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
    plt.savefig(outputdir + f'plot_peaks_{sample_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()   
    
#%% ###############################################################################
# File handling 

def analyze_files(image_paths, outputdir, desired_channel=0, topN=30, min_distance=4, makeplots=True):
    '''
    Loop over given list of image files and perform analysis.
    '''
    # desired_channel=0; topN=30; min_distance=4; makeplots=True
    
    # initialize dataframe list to gather output (on df per image, concatenated later)
    df_list = [None] * len(image_paths)
    
    for idx, img_path in enumerate(image_paths):
        # img_path = image_paths[0]
        
        # determine the filename (for annotation purposes)
        filename = img_path.split('/')[-1].split('.')[0]
        
        # show progress to user
        print(f'Processing image {idx+1}/{len(image_paths)}: {filename}')
        
        # load the image (take only the desired channel)
        img = tifffile.imread(img_path)[desired_channel]
        
        # perform the convolution
        img_conv = convolve_with_disk(img)
        
        # determine the coordinates of max and their values
        coordinates_top_n, topN_vals_conv = get_top_n_local_maxima(img_conv, n=topN, min_distance=min_distance)
        
        # now also get the values of the pixels in the original non-convolved image
        topN_vals_original = img[coordinates_top_n[:, 0], coordinates_top_n[:, 1]]
        
        # estimate a background level, both using mode and above function
        background_lvl_median = estimate_background_level(img)
        background_lvl_mode   = mode(img.ravel()).mode
        
        # now put all information in a small dataframe
        df_list[idx] = pd.DataFrame({
            'filename': filename,
            'coordX': coordinates_top_n[:, 1],
            'coordY': coordinates_top_n[:, 0],
            'filepathfull': img_path,
            'topN_vals_conv': topN_vals_conv,
            'topN_vals_original': topN_vals_original,
            'background_lvl_median': background_lvl_median,
            'background_lvl_mode': background_lvl_mode
        })
        
        # create a plot
        if makeplots:
            os.makedirs(outputdir+'/sample_plots/', exist_ok=True)
            plot_peaks_on_images(img_conv, img, coordinates_top_n, filename, outputdir+'/sample_plots/')
    
    # now concatenate all dfs together
    df_nucleidata = pd.concat(df_list, ignore_index=True)
    
    return df_nucleidata

#%% ###############################################################################
# Plotting functions for the final output

def annotate_df_wfilenames(df_nucleidata, metadata_table=None):
    
    # split the column filename into three new columns, based on splitting by "_"
    df_nucleidata[['KPM', 'replicate', 'growthstadium','tissuetype']] = df_nucleidata['filename'].str.split('_', expand=True).iloc[:, 0:4]
    
    # add the metadata table if present
    if metadata_table is not None:
        # merge the metadata table with the df_nucleidata
        metadata_table['KPM'] = metadata_table['KPM'].astype(str)
        df_nucleidata = pd.merge(df_nucleidata, metadata_table, on='KPM', how='left')
    
    return df_nucleidata
    

def plot_statistics(df_nucleidata, outputdir, showplot=True):
    # now plot the gathered statistics using seaborn
    
    os.makedirs(outputdir+'/summary_plots/', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15.2*cm_to_inch, 15.2*cm_to_inch))
    plt.rcParams.update({'font.size': 8})
    # sns.boxplot(x='label', y='vals', data=df_intensities)    
    sns.violinplot(x='KPM', y='topN_vals_original', data=df_nucleidata, inner=None, facecolor='lightgrey', linewidth=0.5, ax=ax, width=0.8, scale='width')
    sns.stripplot(x='KPM', y='topN_vals_original', data=df_nucleidata, color='black', ax=ax, size=1, hue='replicate', legend=False)
    # rotate x axis labels 90 deg
    plt.xticks(rotation=90)
    # add the level of saturated pixels as a horizontal line
    # ax.axhline(np.iinfo(img1.dtype).max, color='red', linestyle='--', label='background level', linewidth=.5)
    # add the background level as horizontal line
    ax.axhline(np.max(df_nucleidata['background_lvl_mode']), color='blue', linestyle='--', label='background level img1', linewidth=.5)
    ax.set_ylabel('Intensities of brightest')
    ax.set_xlabel('')
    plt.tight_layout()
    # plt.show()
    # save the image to outpudir
    plt.savefig(outputdir+'/summary_plots/' + 'Nuclei_intensities_overview.pdf', dpi=300, bbox_inches='tight')
    if showplot:
        plt.show()
    plt.close()


