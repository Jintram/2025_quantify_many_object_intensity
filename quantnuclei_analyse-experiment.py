



#%% ###############################################################################

import glob

import pandas as pd

# import the script quantnuclei_functions.py, similar to any library
import quantnuclei_functions as qf # import importlib; importlib.reload(qf)

#%% ###############################################################################

# Analysis of experiment "Alldata_202506" 
# Where to look for images
input_images_searchstring = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_KevinPeek/DATA/Alldata_202506/*.tif'
# Location of metadata table (should have column "KPM" which contains a number only, without "KPM")
metadata_filepath = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_KevinPeek/DATA/Excel identifiers martijn.xlsx'
# set the output directory
outputdir = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_KevinPeek/ANALYSIS/20250704_Analysis-Python/'

# Get filelist and metadata table
metadata_table = pd.read_excel(metadata_filepath)
image_paths = glob.glob(input_images_searchstring)

# call the script
df_nucleidata = qf.analyze_files(image_paths, outputdir, desired_channel=0, topN=30, min_distance=4, makeplots=True)

# add more annotation to data
df_nucleidata = qf.annotate_df_wfilenames(df_nucleidata)

# save the processed data to excel file and pkl file
df_nucleidata.to_excel(outputdir + 'Nuclei_intensities.xlsx', index=False)
df_nucleidata.to_pickle(outputdir + 'Nuclei_intensities.pkl')
# data can now be loaded as follows (use this when you want to make a new plot but
# not re-run the whole analysis)
# df_nucleidata = pd.read_pickle(outputdir + 'Nuclei_intensities.pkl')

# generate an overview plot
qf.plot_statistics(df_nucleidata, outputdir, showplot=True)

# %%
