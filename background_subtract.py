def load_im(image,array):
	ii=0
	hdulist = pyfits.open(image)
	for hdu in hdulist:
		im_dim=(4094, 2096) #hdu.shape
		im_ndim=len(im_dim)
		if im_ndim==2:
			array[0,ii,:,:]=hdu.data
			ii+=1
			print ii
	hdulist.close()
	return array

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

#Open image file, and scale it by the weight map SHOULD THE IMAGE BE SCALED????
hdulist = pyfits.open(test_image)
im_nchip=0
for hdu in hdulist:
#	print hdu.header['NAXIS']
	if hdu.header['NAXIS'] != 0:
		hdu = np.array(hdu)
		im_dim= (4094, 2046) #hdu.shape
		print im_dim
		im_ndim=len(im_dim)
		print im_ndim
		if im_ndim==2:
			if im_nchip==0: im_size=im_dim
			im_nchip+=1
hdulist.close()

im_data = np.zeros((1,im_nchip,im_size[0],im_size[1]))
print im_data
im_data = load_im(test_image,im_data)
print im_data
exit()
#weight_data = im_data



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

# im_file=['fornax_t1_d1_g.fits','fornax_t1_d2_g.fits','fornax_t1_d3_g.fits']
# 
# hdulist = pyfits.open(im_file[0])
# im_nchip=0
# for hdu in hdulist:
#   im_dim=hdu.shape
#   im_ndim=len(im_dim)
#   if im_ndim==2:
#     if im_nchip==0: im_size=im_dim
#     im_nchip+=1
# hdulist.close()
# 
# im_data_cube=np.zeros( (len(im_file), im_nchip, im_size[0], im_size[1]) )
# 
# for i in range(len(im_file)):
#   j=0
#   hdulist = pyfits.open(im_file[i])
#   for hdu in hdulist:
#     im_dim=hdu.shape
#     im_ndim=len(im_dim)
#     if im_ndim==2:
#       im_data_cube[i,j,:,:]=hdu.data
#       j+=1
#   hdulist.close()

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