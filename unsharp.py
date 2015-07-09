import pyfits
import numpy as np
import scipy.signal as sig
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import sys
from datetime import datetime

start_time = datetime.now()

prefixin = '/Volumes/MyPassport/SCABSworkbench/'
prefixout = '/Volumes/MyPassport/SCABSworkbench/'

image_file = '%ssurvey_tile1_i_short.fits' % prefixin
mask_file = '%ssurvey_tile1_i_short.MASK.fits' % prefixin

if len(sys.argv) < 2:
	print "ERROR - missing cutout type"
	exit()

if (sys.argv[1] == 'cutout') != True:
	if (sys.argv[1] == 'full') != True:
		print "\nERROR - 'cutout' or 'full' expected for first argument"

if sys.argv[1] == 'cutout':

	header = pyfits.getheader('%ssurvey_tile1_i_short.fits' % prefixin)

	box = 8000
	x = 15650#+box
	y = 14500#-box

	#Cutout the original and mask images
	print "\nCreating image cutout..."
	filein = pyfits.open(image_file)
	cutout = filein[0].section[int(y-0.5*box):int(y+0.5*box),int(x-0.5*box):int(x+0.5*box)]
	print "\nWriting image cutout..."
	pyfits.writeto('%scutout.fits' % prefixout,cutout,header,clobber=True)
	filein.close()

	print "\nCreating mask cutout..."
	mask_filein = pyfits.open(mask_file)
	mask_cutout = mask_filein[0].section[int(y-0.5*box):int(y+0.5*box),int(x-0.5*box):int(x+0.5*box)]
	print "\nWriting mask cutout..."
	pyfits.writeto('%smask_cutout.fits' % prefixout,mask_cutout,header,clobber=True)
	mask_filein.close()

if sys.argv[1] == 'full':
	header = pyfits.getheader(image_file)	
	print "\nReading image file..."
	filein = pyfits.open(image_file)
	cutout = filein[0].data
	mask_header = pyfits.getheader(mask_file)
	print "\nReading image mask..."
	mask_filein = pyfits.open(mask_file)
	mask_cutout = mask_filein[0].data

#Create the original mask
print "\nMasking image..."
nosource = cutout
bv_mask = (mask_cutout == 1.)
nosource[bv_mask]=np.nan
print "Writing masked image..."
pyfits.writeto('%snosource.fits' % prefixout,nosource,header,clobber=True)

#Blur the source-free original
filt_size = 25
print "Blurring image with %i-pixel median filter..." % filt_size
mask = nd.median_filter(nosource,filt_size)
print "Done."

#Write the blurred image
print "\nWriting blurred image..."
pyfits.writeto('%smask.fits' % prefixout,mask,header,clobber=True)
w = 1.
print "\nSubtracting blurred image from original cutout..."
unsharp = nosource-mask*w
#Write the unsharp mask
print "\nWriting unsharp-masked image..."
pyfits.writeto('%sunsharp.fits' % prefixout,unsharp,header,clobber=True)
print "Done in %5.2f minutes." % ((datetime.now() - startime)/60.)