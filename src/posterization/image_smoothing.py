#!/usr/bin/env python3

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def get_median_color( window ):
    
    channel1 = window[:, :, 0]
    channel2 = window[:, :, 1]
    channel3 = window[:, :, 2]
    
    color = np.zeros( (  3 ), dtype = np.uint8 )
    
    color[0] = np.median( channel1 )
    color[1] = np.median( channel2 )
    color[2] = np.median( channel3 )
    
    return color

@jit(nopython=True)
def medianBlur_truncate( img, dft_img, window_size ):
    margin = int( ( window_size - 1 ) / 2 )

    filtered_img = np.copy( img )
    for i in range( margin, img.shape[0] - margin ):
        for j in range( margin, img.shape[1] - margin ):

            if dft_img[i, j] < 0.1:
                window = img[i - margin: i + margin + 1, j - margin: j + margin + 1]
                
                median_color = get_median_color( window )
                filtered_img[i, j] = median_color
                
    print( 'Blurred!')
    return filtered_img
        
def main():
    import argparse
    parser = argparse.ArgumentParser( description = 'Smooth posterized image.' )
    parser.add_argument( 'input_image', help = 'The path to the input image.' )
    parser.add_argument( 'output_path', help = 'Where to save the output smoothed image.' )
    args = parser.parse_args()
        
        
    img = cv2.imread( args.input_image )
    img_og = np.copy( img )
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # algorithm starts
    start = time.time()
    
    # DFT and inverse DFT
    rows, cols = gray.shape
    crow, ccol = int( rows / 2 ) , int( cols / 2 )
    f = np.fft.fft2( gray )
    fshift = np.fft.fftshift( f )
    cv2.imwrite( args.output_path + '-mag_spec.jpg', 20 * np.log( np.abs( fshift ) ) )
    
    
    
    fshift[crow - 25: crow + 25, ccol - 25: ccol + 25] = 0
    f_ishift = np.fft.ifftshift( fshift )
    img_back = np.fft.ifft2( f_ishift )
    img_back = np.abs( img_back ) / 255.    # black: 0, white: 1
    
    frq = np.sort( np.copy( img_back ).ravel() )
    plt.plot( frq )
    plt.ylabel( 'grayscale (normalized)' )
    plt.savefig( args.output_path + '-frequency.jpg' )
    
    
    img = medianBlur_truncate( img, img_back, 7 )
    img = medianBlur_truncate( img, img_back, 7 )
    img = medianBlur_truncate( img, img_back, 7 )
    
    
    img_og = cv2.medianBlur( img_og, 7 )
    img_og = cv2.medianBlur( img_og, 7 )
    img_og = cv2.medianBlur( img_og, 7 )
    
    end = time.time()
    print( "Finished. Total time: ", end - start )
    
    
    cv2.imwrite( args.output_path + '-gray.jpg', gray )
    cv2.imwrite( args.output_path + '-DFT.jpg', img_back * 255. )
    cv2.imwrite( args.output_path +  '-pyMF.jpg', img_og )
    cv2.imwrite( args.output_path +  '.jpg', img )


if __name__ == '__main__':
    main()

