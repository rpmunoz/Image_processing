def load_im(image,array):
	jj=0
	hdulist = pyfits.open(image)
	for hdu in hdulist:
	  if hdu.header['NAXIS'] != 0:
	    im_dim=(4094, 2096) #hdu.shape
	    im_ndim=len(im_dim)
	    if im_ndim==2:
		  array[0,jj,:,:]=hdu.data
		  jj+=1
	hdulist.close()
	return array

def find_nearest(array,nvals,val):
	sorted = np.sort(np.abs(array))
	keep_vals = sorted[0:nvals]
	inds = []
	for value in keep_vals:
		inds.append(list(np.abs(array)).index(value))
	inds = np.array(inds)
	return inds

def median_subtract(back_cube,med_data,image,chip):
	for ii in range(im_size[0]):
	  for jj in range(im_size[1]):
		back_vals = []
		for kk in range(n_backs):
		  back_vals.append(back_cube[kk][chip][ii][jj])
		med_data[0][chip][ii][jj] = np.median(back_vals)
	image[0][chip] = image[0][chip] - med_data[0][chip]
	return image, med_data

import pyfits
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
from datetime import datetime

start_time = datetime.now()

if len(sys.argv) < 2 :
	print "ERROR - missing subtraction method"
	exit()
if sys.argv[1] != "median":
	print "ERROR - background subtraction method must be one of [median]"
	exit()

im_dir = '/Volumes/Q6/matt/2014A-0610/background_test_files/' #'/Volumes/Q6/matt/2014A-0610/pipeline/images/'
do_tile = '1'
do_dither = '1'
do_filt = 'i'
n_backs = 3

#back_dir = '/Volumes/MyPassport/masks/'
#mask_dir = '/Volumes/MyPassport/masks/'
back_dir = '/Volumes/Q6/matt/2014A-0610/background_test_files/' #'/Volumes/Q6/matt/2014A-0610/pipeline/images/'
mask_dir = '/Volumes/Q6/matt/2014A-0610/background_test_files/' #'/Volumes/Q6/matt/2014A-0610/pipeline/images/'

test_image = im_dir+'survey_t'+do_tile+'_d'+do_dither+'_'+do_filt+'_short.fits'
test_weight = im_dir+'survey_t'+do_tile+'_d'+do_dither+'_'+do_filt+'_short.WEIGHT.fits'

im_h = pyfits.open(test_image)[0].header

#Open image file, and scale it by the weight map SHOULD THE IMAGE BE SCALED????
print "\nLoading image..."
hdulist = pyfits.open(test_image)
im_nchip=0
for hdu in hdulist:
	if hdu.header['NAXIS'] != 0:
		hdu = np.array(hdu)
		im_dim= (4094, 2046) #hdu.shape
		im_ndim=len(im_dim)
		if im_ndim==2:
			if im_nchip==0: im_size=im_dim
			im_nchip+=1
hdulist.close()

im_data = np.zeros((1,im_nchip,im_size[0],im_size[1]))
#print im_data
im_data = load_im(test_image,im_data)
#print im_data
#weight_data = im_data

# x = np.arange(0,4094,1)
# y = np.arange(0,2046,1)
# x, y = np.meshgrid(x,y)
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# ax.plot_surface(x,y,im_data[0][0])
# plt.show()
# exit()

####################################################################
####Open background and mask files and scale them by weight maps####
####################################################################

#Determine the background files closest in time to the image
temp_back_ims = glob.glob(back_dir+'survey_t*_*d'+do_dither+'_*'+do_filt+'_*short.fits')
temp_back_weights = glob.glob(back_dir+'survey_t*_*d'+do_dither+'_*'+do_filt+'_*short.WEIGHT.fits')

temp_mask_ims = glob.glob(mask_dir+'survey_t*_*d'+do_dither+'_*'+do_filt+'_*short.MASK.fits')

# print temp_back_ims
# print temp_back_weights
# print temp_mask_ims

im_mjd = im_h['MJD-OBS']
print "\nImage observation date = %f" % im_mjd

temp_back_ims = np.delete(temp_back_ims, list(temp_back_ims).index(back_dir+'survey_t1_'+'d'+do_dither+'_'+do_filt+'_short.fits'),0)
temp_back_weights = np.delete(temp_back_weights, list(temp_back_weights).index(back_dir+'survey_t1_'+'d'+do_dither+'_'+do_filt+'_short.WEIGHT.fits'), 0)
temp_mask_ims = np.delete(temp_mask_ims, list(temp_mask_ims).index(back_dir+'survey_t1_'+'d'+do_dither+'_'+do_filt+'_short.MASK.fits'), 0)

# print temp_back_ims
# print temp_back_weights

back_mjds = []
for ii in range(len(temp_back_ims)):
	hdulist = pyfits.open(temp_back_ims[ii])
	back_mjds.append(float(hdulist[0].header['MJD-OBS']))
back_mjds = np.array(back_mjds)

time_diffs = back_mjds - im_mjd
#print len(temp_back_ims), len(temp_back_weights), len(temp_mask_ims), len(time_diffs)
#print time_diffs

back_ims = []
back_weights = []
mask_ims = []

for idx in find_nearest(time_diffs,n_backs,0):
	back_ims.append(temp_back_ims[idx])
	back_weights.append(temp_back_weights[idx])
	mask_ims.append(temp_mask_ims[idx])

back_ims = np.array(back_ims)
back_weights = np.array(back_weights)
mask_ims = np.array(mask_ims)

# print back_ims
# print back_weights
# print mask_ims

print "\nUsing the following frames for the background subtraction: " 
print back_ims

back_mjds = []
for ii in range(n_backs):
	hdulist = pyfits.open(back_ims[ii])
	back_mjds.append(float(hdulist[0].header['MJD-OBS']))

#Load background images and masks
back_im_cube = np.zeros( (len(back_ims), im_nchip, im_size[0], im_size[1]) )
for ii in range(len(back_ims)):
	print "\nLoading background image %i/%i into data cube..." % (ii+1,len(back_ims))
	back_im_temp = np.zeros( (1, im_nchip, im_size[0], im_size[1]) )
	back_im_cube[ii] = load_im(back_ims[ii],back_im_temp)

#print back_im_cube
#print len(back_im_cube), len(back_im_cube[0]), len(back_im_cube[0][0]), len(back_im_cube[0][0][0])

mask_im_cube = np.zeros( (len(mask_ims), im_nchip, im_size[0], im_size[1]) )
for ii in range(len(mask_ims)):
	print "\nLoading mask image %i/%i into data cube..." % (ii+1,len(mask_ims))
	mask_im_temp = np.zeros( (1, im_nchip, im_size[0], im_size[1]) )
	mask_im_cube[ii] = load_im(mask_ims[ii],mask_im_temp)

#Mask background images
#print np.median(back_im_cube[0][0])

for ii in range(len(mask_ims)):
	print "\nMasking background image %i/%i..." % (ii+1,len(mask_ims))
	for jj in range(len(back_im_cube[0])):
		#print "Median of Chip #%i before masking: %7.3f" % (jj+1, np.median(back_im_cube[ii][jj]))
		bv_mask = (mask_im_cube[ii][jj] == 1.)
		back_im_cube[ii][jj][bv_mask] = np.nan
		#print "Median of Chip #%i after masking: %7.3f" % (jj+1, np.median(back_im_cube[ii][jj]))

#print back_im_cube[0][0][0][0], back_im_cube[1][0][0][0], back_im_cube[2][0][0][0]
#print back_im_cube[0][0][100][0], back_im_cube[1][0][100][0], back_im_cube[2][0][100][0]

########################################################################
####Calculate the background level from all of the background images####
########################################################################

print im_data[0][0][0][0]

#Median background subtraction
if sys.argv[1] == 'median':
	for chip_num in range(im_nchip):
		med_back_data = np.zeros( (1, im_nchip, im_size[0], im_size[1]) )
		im_data, med_back_data = median_subtract(back_im_cube,med_back_data,im_data,chip_num)

#print med_back_data[0][0][0][0]
#print im_data[0][0][0][0]

print datetime.now(), start_time
print "Done in %5.2f minutes." % ((datetime.now() - start_time)/60.)

###########################################################################
####Fit a spline to the surface of each chip, for each background image####
###########################################################################

