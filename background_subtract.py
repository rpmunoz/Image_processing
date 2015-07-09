def open_im(image):
	hdulist = pyfits.open(image)
	temp_image_data = hdulist[1].data
	temp_image_h = hdulist[0].header
	hdulist.close()
	
	return temp_image_data, temp_image_h

def weight_scale(im,weight):
	temp_im_data, temp_im_h = open_im(im)
	temp_w_data, temp_w_h = open_im(weight)
	
	bv_weight = (temp_w_data == 0.)
	temp_im_data[bv_weight] = np.nan
	
	return temp_im_data, temp_im_h

def mask_image(im,mask):
	temp_im_data, temp_im_h = open_im(im)
	temp_mask_data, temp_mask_h = open_im(mask)
	
	bv_mask = (temp_mask_data == 1)
	temp_im_data[bv_mask] = np.nan
	
	return temp_im_data

def find_nearest(array,nvals,val):
	sorted = np.sort(np.abs(array))
	keep_vals = sorted[0:nvals]
	inds = []
	for value in keep_vals:
		inds.append(list(np.abs(array)).index(value))
	inds = np.array(inds)
	return inds

import pyfits
import numpy as np
import glob

im_dir = '/Volumes/MyPassport/masks/'
do_tile = '1'
do_dither = '1'
do_filt = 'i'
n_backs = 2

back_dir = '/Volumes/MyPassport/masks/'
mask_dir = '/Volumes/MyPassport/masks/'

test_image = im_dir+'survey_t'+do_tile+'_d'+do_dither+'_'+do_filt+'_short.fits'
test_weight = im_dir+'survey_t'+do_tile+'_d'+do_dither+'_'+do_filt+'_short.WEIGHT.fits'

#Open image file, and scale it by the weight map
im_data, im_h = weight_scale(test_image,test_weight)

####Open background and mask files and scale them by weight maps
#Determine the background files closest in time to the image
temp_back_ims = glob.glob(back_dir+'survey_t*_*d'+do_dither+'_*'+do_filt+'_*short.fits')
temp_back_weights = glob.glob(back_dir+'survey_t*_*d'+do_dither+'_*'+do_filt+'_*short.WEIGHT.fits')

temp_mask_ims = glob.glob(mask_dir+'survey_t*_*d'+do_dither+'_*'+do_filt+'_*short.MASK.fits')

print temp_back_ims
print temp_back_weights
print temp_mask_ims

im_mjd = im_h['MJD-OBS']
print "\nImage observation date = %f" % im_mjd

temp_back_ims = np.delete(temp_back_ims, list(temp_back_ims).index(back_dir+'survey_t1_'+'d'+do_dither+'_'+do_filt+'_short.fits'),0)
temp_back_weights = np.delete(temp_back_weights, list(temp_back_weights).index(back_dir+'survey_t1_'+'d'+do_dither+'_'+do_filt+'_short.WEIGHT.fits'), 0)

print temp_back_ims
print temp_back_weights

back_mjds = []
for ii in range(len(temp_back_ims)):
	hdulist = pyfits.open(temp_back_ims[ii])
	back_mjds.append(float(hdulist[0].header['MJD-OBS']))
back_mjds = np.array(back_mjds)

time_diffs = back_mjds - im_mjd
print time_diffs

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

print back_ims
print back_weights
print mask_ims

#Scale and mask the background images
####NEED TO CREATE THE CUBE HERE
for ii in range(len(back_ims)):
	if back_ims[0] == back_weights[0].replace('WEIGHT.','') and back_ims[0] == mask_ims[0].replace('MASK.',''):
		print back_ims[ii], mask_ims[ii]
		back_ims[ii], back_h = weight_scale(back_ims[ii],back_weights[ii])
		back_ims[ii] = mask_image(back_ims[ii],mask_ims[ii])

exit()
back_ims = ['survey_t2_d1_i_short.fits','survey_t3_d1_i_short.fits','survey_t4_d1_i_short.fits']
back_weights = ['survey_t2_d1_i_short.WEIGHT.fits','survey_t3_d1_i_short.WEIGHT.fits','survey_t4_d1_i_short.WEIGHT.fits']

#Mask the background images

#Create data cube of background images

#Iterate through each chip with the following:
#For chip xx, fit a spline on each background image and calculate the median of each pixel.
#Save the medians as a new image.

#Subtract the median-splines from the image file

#Save the background subtracted image.