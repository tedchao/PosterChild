#!/usr/bin/env python3

import sys
from pathlib import Path
from .posterization_gui import *
from .simplepalettes import *

import numpy as np
import cv2
import gco

try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PySide2.QtCore import *
    from PySide2.QtGui import *
    from PySide2.QtWidgets import *
    
from PIL import Image

# In command line: "pip install opencv-python-headless" to avoid qt complaining two set of binaries

### This sometimes happens: 'qt.gui.icc: fromIccProfile: failed minimal tag size sanity'

class TimerMessageBox( QMessageBox ):
    def __init__( self, timeout = 3, parent = None ):
        super( TimerMessageBox, self ).__init__( parent )
        self.setWindowTitle( "Algorithm is processing your image. Please hold on." )
        self.time_to_wait = timeout
        self.setText( "Algorithm is processing your image. Please hold on." )
        self.setStandardButtons( QMessageBox.NoButton )
        self.timer = QTimer()
        self.timer.setInterval( 100 )
        self.timer.timeout.connect( self.changeContent )
        self.timer.start()
        
    def changeContent( self ):
        self.time_to_wait -= 1
        if self.time_to_wait <= 0:
            self.close()
            
    def closeEvent( self, event ):
        self.timer.stop()
        event.accept()


class MainWindow( QWidget ):
    
    def __init__( self ):
        super().__init__()
        
        self.title = 'Posterization'
        self.x = 300
        self.y = 0
        self.width = 100
        self.height = 300
        
        self.initUI()
    
    def initUI( self ):
        self.setWindowTitle( self.title )
        self.setGeometry( self.x, self.y, self.width, self.height )
        #self.setStyleSheet("background-color: white;") 
        
        # Set the welcome icon in GIF
        self.gif = QMovie( str( Path( __file__ ).parent / "simpson.gif" ) )
        self.imageLabel = QLabel()
        self.imageLabel.setMovie( self.gif )
        self.gif.start()
        
        ### palette images
        self.paletteLabel = QLabel()
        self.paletteLabel.setPixmap( QPixmap() )
        
        #### Variables
        self.imagePath = ""
        self.palette = None
        
        self.show_palette = 0
        self.show_input = 0
        
        self.blur_window_slider_val = 7 # default
        self.blur_slider_val = 0.1      # default
        self.binary_slider_val = 1.0    # default
        self.cluster_slider_val = 20    # default
        self.palette_slider_val = 6     # default
        self.blend_slider_val = 3       # default
        self.current_image_indx = -1    # Track the current image index in the image list
        
        self.imageList = []
        self.paletteList = [-1 * np.ones( ( 1, 1, 1 ) )]
        
        self.input_image = None         # Store it as np array
        self.saliency_map = None
        self.posterized_image_wo_smooth = -1 * np.ones( ( 1, 1, 1 ) )      # Store it as np array
        self.posterized_image_w_smooth = -1 * np.ones( ( 1, 1, 1 ) )       # Store it as np array
        
        
        #### BOXES
        btns_io_box = QVBoxLayout() # set bottons' box for I/O
        
        # algorithm_btns_box = QVBoxLayout() # set bottons' box for algorithms
        sld_box_palette = QHBoxLayout()
        sld_box_blend = QHBoxLayout()
        sld_box_cluster = QHBoxLayout()
        sld_box_binary = QHBoxLayout()
        sld_box_blur = QHBoxLayout()
        sld_box_window = QHBoxLayout()
        
        blur_box = QHBoxLayout()
        
        img_box = QHBoxLayout() # set image's box
        pages_box = QVBoxLayout() # set next-previous box
        show_hide_box = QVBoxLayout()
        
        
        #### BUTTONS
        # button for selecting an input image
        self.img_btn = QPushButton( 'Choose image' )
        self.img_btn.clicked.connect( self.get_image )
        self.img_btn.setToolTip( 'Press the button to <b>select</b> an image.' ) 
        self.img_btn.setMaximumWidth( 150 )
        
        # button for posterizing the given image
        self.posterize_btn = QPushButton( 'Posterize!' )
        self.posterize_btn.clicked.connect( self.posterize )
        self.posterize_btn.setToolTip( 'Press the button to <b>posterize</b> your image.' ) 
        self.posterize_btn.setMaximumWidth( 150 )
        
        # button for re-smoothing the posterized image
        self.smooth_btn = QPushButton( 'Re-smooth' )
        self.smooth_btn.clicked.connect( self.smooth )
        self.smooth_btn.setToolTip( 'Press the button to <b>re-smooth</b> your posterized image.' ) 
        self.smooth_btn.setMaximumWidth( 150 )
        
        # button for loading the saliency map
        self.map_btn = QPushButton( 'Smooth with custom map' )
        self.map_btn.clicked.connect( self.pop_up_load_saliency_map )
        self.map_btn.setToolTip( 'Press the button to <b>load</b> your own map to blur.' ) 
        self.map_btn.setMaximumWidth( 180 )

        # button for saving the posterized image
        self.save_btn = QPushButton( 'Save current image' )
        self.save_btn.clicked.connect( self.save_current_image )
        self.save_btn.setToolTip( 'Press the button to <b>save</b> your current image.' ) 
        self.save_btn.setMaximumWidth( 150 )
        
        # button for saving the palette
        self.save_palette_btn = QPushButton( 'Save palette' )
        self.save_palette_btn.clicked.connect( self.save_current_palette )
        self.save_palette_btn.setToolTip( 'Press the button to <b>save</b> the palette.' ) 
        self.save_palette_btn.setMaximumWidth( 150 )
        
        
        #### Previous-next buttons
        self.previous_btn = QPushButton( '<<<-- See History' )
        self.previous_btn.clicked.connect( self.paste_previous_image )
        self.previous_btn.setToolTip( 'Press the button to see your <b>previous</b> image in the gallory.' ) 
        self.previous_btn.setMinimumWidth( 165 )
        self.previous_btn.setMaximumWidth( 165 )
        
        self.next_btn = QPushButton( 'See Next -->>>' )
        self.next_btn.clicked.connect( self.paste_next_image )
        self.next_btn.setToolTip( 'Press the button to see your <b>next</b> image in the gallory.' ) 
        self.next_btn.setMinimumWidth( 165 )
        self.next_btn.setMaximumWidth( 165 )
        
        #### Show/Hide buttons
        self.palette_btn = QPushButton( 'Show/Hide palette' )
        self.palette_btn.clicked.connect( self.show_hide_palette )
        self.palette_btn.setToolTip( 'Press the button to <b>show</b> or <b>hide</b> the palette.' ) 
        self.palette_btn.setMinimumWidth( 165 )
        self.palette_btn.setMaximumWidth( 165 )
        
        self.og_img_btn = QPushButton( 'Show/Hide input image' )
        self.og_img_btn.clicked.connect( self.show_hide_input_image )
        self.og_img_btn.setToolTip( 'Press the button to <b>show</b> your input image.' ) 
        self.og_img_btn.setMinimumWidth( 165 )
        self.og_img_btn.setMaximumWidth( 165 )
        
        
        #### SLIDERS
        # slider for palette size
        self.blend_sld = QSlider( Qt.Horizontal )
        self.blend_sld.setRange( 1, 25 )
        self.blend_sld.setFocusPolicy( Qt.NoFocus )
        self.blend_sld.setSliderPosition( self.blend_slider_val )
        self.blend_sld.setPageStep( 1 )
        self.blend_sld.setToolTip( 'Fine-tune the slider to get your desired main palette size.' ) 
        self.blend_sld.setMinimumWidth( 150 )
        self.blend_sld.setMaximumWidth( 200 )
        self.blend_sld.valueChanged.connect( self.blend_change_slider )
        
        # slider for palette size
        self.palette_sld = QSlider( Qt.Horizontal )
        self.palette_sld.setRange( 4, 15 )
        self.palette_sld.setFocusPolicy( Qt.NoFocus )
        self.palette_sld.setSliderPosition( self.palette_slider_val )
        self.palette_sld.setPageStep( 1 )
        self.palette_sld.setToolTip( 'Fine-tune the slider to get your desired main palette size.' ) 
        self.palette_sld.setMinimumWidth( 150 )
        self.palette_sld.setMaximumWidth( 200 )
        self.palette_sld.valueChanged.connect( self.palette_change_slider )
        
        # slider for number of clusters for Kmeans
        self.cluster_sld = QSlider( Qt.Horizontal )
        self.cluster_sld.setRange( 15, 50 )
        self.cluster_sld.setFocusPolicy( Qt.NoFocus )
        self.cluster_sld.setSliderPosition( self.cluster_slider_val )
        self.cluster_sld.setPageStep( 1 )
        self.cluster_sld.setToolTip( 'Fine-tune the slider to get your desired threshold for outlier colors.' ) 
        self.cluster_sld.setMinimumWidth( 150 )
        self.cluster_sld.setMaximumWidth( 200 )
        self.cluster_sld.valueChanged.connect( self.cluster_change_slider )
        
        # slider for binary penalization
        self.binary_sld = QSlider( Qt.Horizontal )
        self.binary_sld.setRange( 1, 200 )
        self.binary_sld.setFocusPolicy( Qt.NoFocus )
        self.binary_sld.setSliderPosition( int( 100 * self.binary_slider_val ) )
        self.binary_sld.setPageStep( 1 )
        self.binary_sld.setToolTip( 'Fine-tune the slider to get your desired penalization on binary term.' ) 
        self.binary_sld.setMinimumWidth( 150 )
        self.binary_sld.setMaximumWidth( 200 )
        self.binary_sld.valueChanged.connect( self.binary_change_slider )
        
        # slider for blurring threshold
        self.blur_sld = QSlider( Qt.Horizontal )
        self.blur_sld.setRange( 0, 100 )
        self.blur_sld.setFocusPolicy( Qt.NoFocus )
        self.blur_sld.setSliderPosition( int( 100 * self.blur_slider_val ) )
        self.blur_sld.setPageStep( 1 )
        self.blur_sld.setToolTip( 'Fine-tune the slider to get your desired blurring threshold.' ) 
        self.blur_sld.setMinimumWidth( 150 )
        self.blur_sld.setMaximumWidth( 200 )
        self.blur_sld.valueChanged.connect( self.blur_change_slider )
        
        # slider for blurring threshold
        self.blur_window_sld = QSlider( Qt.Horizontal )
        self.blur_window_sld.setRange( 0, 3 )
        self.blur_window_sld.setFocusPolicy( Qt.NoFocus )
        self.blur_window_sld.setSliderPosition( ( self.blur_window_slider_val - 3 ) / 2 )
        self.blur_window_sld.setPageStep( 1 )
        self.blur_window_sld.setToolTip( 'Fine-tune the slider to get your desired blurring window size.' ) 
        self.blur_window_sld.setMinimumWidth( 150 )
        self.blur_window_sld.setMaximumWidth( 200 )
        self.blur_window_sld.valueChanged.connect( self.blur_window_change_slider )
        
        
        ### LABELS
        # labels
        self.blur_window_text = QLabel( 'Boundary smoothess (Default: 7):' )
        self.blur_text = QLabel( 'Detail threshold (Default: 0.1):' )
        self.binary_text = QLabel( 'Region clumpiness (Default: 1.0):' )
        self.cluster_text = QLabel( 'Outlier removable strength (Default: 20):' )
        self.palette_text = QLabel( 'Palette size (Default: 6):' )
        self.blend_text = QLabel( 'Palette blends (Default: 3):' )
        
        self.blur_window_text.setMaximumWidth( 250 )
        self.blur_text.setMaximumWidth( 250 )
        self.binary_text.setMaximumWidth( 250 )
        self.cluster_text.setMaximumWidth( 250 )
        self.palette_text.setMaximumWidth( 250 )
        self.blend_text.setMaximumWidth( 250 )
        
        # label text for blur slider
        self.blur_window_sld_label = QLabel( '7' )
        self.blur_window_sld_label.setAlignment( Qt.AlignLeft )
        self.blur_window_sld_label.setMinimumWidth( 80 )
        
        self.blur_sld_label = QLabel( '0.1' )
        self.blur_sld_label.setAlignment( Qt.AlignLeft )
        self.blur_sld_label.setMinimumWidth( 80 )
        
        # label text for binary penalization slider
        self.binary_sld_label = QLabel( '1.0' )
        self.binary_sld_label.setAlignment( Qt.AlignLeft )
        self.binary_sld_label.setMinimumWidth( 80 )
        
        # label text for kmeans cluster slider
        self.cluster_sld_label = QLabel( '20' )
        self.cluster_sld_label.setAlignment( Qt.AlignLeft )
        self.cluster_sld_label.setMinimumWidth( 80 )
        
        # label text for palette size slider
        self.palette_sld_label = QLabel( '6' )
        self.palette_sld_label.setAlignment( Qt.AlignLeft )
        self.palette_sld_label.setMinimumWidth( 80 )
        
        # label text for blending way slider
        self.blend_sld_label = QLabel( '3' )
        self.blend_sld_label.setAlignment( Qt.AlignLeft )
        self.blend_sld_label.setMinimumWidth( 80 )
        
        
        ### BOX FRAMES
        btns_io_box.addStretch(20)
        btns_io_box.addWidget( self.img_btn )
        btns_io_box.addWidget( self.save_btn )
        btns_io_box.addWidget( self.save_palette_btn )
        btns_io_box.addStretch(20)
        
        
        # Separate boxes for parameters
        sld_box_palette.addStretch(20)
        sld_box_palette.addWidget( self.palette_sld )
        sld_box_palette.addWidget( self.palette_sld_label )
        sld_box_palette.addStretch(20)
        
        sld_box_blend.addStretch(2)
        sld_box_blend.addWidget( self.blend_sld )
        sld_box_blend.addWidget( self.blend_sld_label )
        sld_box_blend.addStretch(2)
        
        sld_box_cluster.addStretch(2)
        sld_box_cluster.addWidget( self.cluster_sld )
        sld_box_cluster.addWidget( self.cluster_sld_label )
        sld_box_cluster.addStretch(2)
        
        sld_box_binary.addStretch(2)
        sld_box_binary.addWidget( self.binary_sld )
        sld_box_binary.addWidget( self.binary_sld_label )
        sld_box_binary.addStretch(2)
        
        sld_box_blur.addStretch(2)
        sld_box_blur.addWidget( self.blur_sld )
        sld_box_blur.addWidget( self.blur_sld_label )
        sld_box_blur.addStretch(2)
        
        sld_box_window.addStretch(2)
        sld_box_window.addWidget( self.blur_window_sld )
        sld_box_window.addWidget( self.blur_window_sld_label )
        sld_box_window.addStretch(2)
        
        # blur box for re-smooth and smooth by map
        blur_box.addStretch(2)
        blur_box.addWidget( self.smooth_btn )
        blur_box.addWidget( self.map_btn )
        blur_box.addStretch(2)
        
        # Image box
        img_box.addStretch(1)
        img_box.addWidget( self.paletteLabel )
        img_box.addStretch(1)
        img_box.addWidget( self.imageLabel )
        img_box.addStretch(4)
        
        # Previous-next box
        pages_box.addWidget( self.previous_btn )
        show_hide_box.addWidget( self.next_btn )
        
        # Show-hide box
        pages_box.addWidget( self.palette_btn )
        show_hide_box.addWidget( self.og_img_btn )
        
        # Set grid layout
        grid = QGridLayout()
        
        grid.addLayout( btns_io_box, 0, 0 )
        
        ### parameters for posterization
        grid.addWidget( self.palette_text, 2, 0 )
        grid.addLayout( sld_box_palette, 3, 0 )
        
        grid.addWidget( self.blend_text, 4, 0 )
        grid.addLayout( sld_box_blend, 5, 0 )
        
        grid.addWidget( self.cluster_text, 6, 0 )
        grid.addLayout( sld_box_cluster, 7, 0 )
        
        grid.addWidget( self.binary_text, 8, 0 )
        grid.addLayout( sld_box_binary, 9, 0 )
        
        grid.addWidget( self.posterize_btn, 10, 0 )
        
        ### parameters for smoothing
        grid.addWidget( self.blur_text, 12, 0 )
        grid.addLayout( sld_box_blur, 13, 0 )
        
        grid.addWidget( self.blur_window_text, 14, 0 )
        grid.addLayout( sld_box_window, 15, 0 )
        
        grid.addLayout( blur_box, 16, 0 )
        
        
        
        grid.addLayout( pages_box, 0, 10 )
        grid.addLayout( show_hide_box, 0, 11 )
        grid.addLayout( img_box, 1, 1, 20, 20 )
        self.setLayout(grid) 
        
        self.show()
    
    # Slider functions
    def blur_window_change_slider(self, value):
        self.blur_window_slider_val = 2 * value + 3
        self.blur_window_sld_label.setText( str( 2 * value + 3 ) )
        
    def blur_change_slider(self, value):
        self.blur_slider_val = value / 100
        self.blur_sld_label.setText( str( value / 100 ) )
    
    def binary_change_slider(self, value):
        self.binary_slider_val = value / 100
        self.binary_sld_label.setText( str( value / 100 ) )
    
    def cluster_change_slider(self, value):
        self.cluster_slider_val = value
        self.cluster_sld_label.setText( str( value ) )
    
    def palette_change_slider(self, value):
        self.palette_slider_val = value
        self.palette_sld_label.setText( str( value ) )
        
    def blend_change_slider(self, value):
        self.blend_slider_val = value
        self.blend_sld_label.setText( str( value ) )
    
    
    # Function for selecting an input image
    def get_image( self ):
        img = QFileDialog.getOpenFileName( self, 'Select file' )
        if img:
            path = img[0]
            self.load_image( path )
        else:
            QMessageBox.warning( self, 'Warning' , 'No file selected.' )
    
    def paste_previous_image( self ):
        self.current_image_indx -= 1
        if self.current_image_indx == -2:
            QMessageBox.warning( self,'Warning','Please select an image first.' )
            self.current_image_indx += 1
            
        elif self.current_image_indx == -1:
            QMessageBox.warning( self,'Warning','No more previous image.' )
            self.current_image_indx += 1
            
        else:
            if self.current_image_indx != 0 and self.show_palette == 1:
                self.set_image( self.paletteLabel, self.paletteList[self.current_image_indx] )
            else:   # input image has no palette, so place a blank
                self.paletteLabel.setPixmap( QPixmap() )
            self.set_image( self.imageLabel, self.imageList[self.current_image_indx] )
        
    def paste_next_image( self ):
        self.current_image_indx += 1
        if self.current_image_indx == 0:
            QMessageBox.warning( self,'Warning','Please select an image first.' )
            self.current_image_indx -= 1
            
        elif self.current_image_indx == len( self.imageList ):
            QMessageBox.warning( self,'Warning','No more next image.' )
            self.current_image_indx -= 1
            
        else:
            if self.current_image_indx != 0 and self.show_palette == 1:
                self.set_image( self.paletteLabel, self.paletteList[self.current_image_indx] )
            else:   # input image has no palette, so place a blank
                self.paletteLabel.setPixmap( QPixmap() )
            self.set_image( self.imageLabel, self.imageList[self.current_image_indx] )
    
    #Load new image function
    def set_image( self, panel, image ):
        #Load the image into the label
        height, width, dim = image.shape
        qim = QImage( image.data, width, height, 3 * width, QImage.Format_RGB888 )
        panel.setPixmap( QPixmap( qim ) )
        ## Update didn't help. Repaint works.
        # panel.update()
        panel.repaint()
    
    def add_to_imageList( self, image ):
        self.imageList.append( np.asarray( image ) )
        
    def add_to_paletteList( self, palette ):
        self.paletteList.append( np.asarray( palette ) )
        
    def load_image( self, path ):
        print ( "Loading Image." )
        self.imageList = []     # initialized back to empty when giving another input image
        
        # push input image in the list
        self.current_image_indx += 1
        self.input_image = cv2.cvtColor( cv2.imread( path ), cv2.COLOR_BGR2RGB )
        self.add_to_imageList( self.input_image )
        
        self.imageLabel.setPixmap( QPixmap( path ) )
        self.imagePath = path
    
    def show_hide_palette( self ):
        if self.imagePath == "":
            QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
            
        elif self.paletteList[-1][0, 0, 0] == -1:
            QMessageBox.warning( self, 'Warning', 'You do not have palette. Please posterize your image first.' )
        else:
            self.show_palette = 1 - self.show_palette
            if self.current_image_indx != 0 and self.show_palette == 1:
                self.set_image( self.paletteLabel, self.paletteList[self.current_image_indx] )
            else:   # input image has no palette, so place a blank
                self.paletteLabel.setPixmap( QPixmap() )
        
    def show_hide_input_image( self ):
        if self.imagePath == "":
            QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
        elif self.imagePath != "" and self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'This is your input image.' )
        else:
            self.show_input = 1 - self.show_input
            if self.show_input == 1:
                self.set_image( self.imageLabel, self.imageList[0] )
            else:
                self.set_image( self.imageLabel, self.imageList[self.current_image_indx] )
    
    # posterization
    def posterize( self ):
        if self.imagePath == "":
            QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
        else:
            print( "Start posterizing." )
            img_arr = np.asfarray( PIL.Image.open( self.imagePath ).convert( 'RGB' ) ) / 255.
            
            # algorithm starts
            start = time.time()
            
            messagebox = TimerMessageBox( 1, self )
            messagebox.open()
                
            # K-means
            img_arr_re = img_arr.reshape( ( -1, 3 ) )
            img_arr_cluster = get_kmeans_cluster_image( self.cluster_slider_val, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
            
            # MLO
            post_img, final_colors, add_mix_layers, palette = \
            posterization( self.imagePath, img_arr, img_arr_cluster, self.palette_slider_val, self.blend_slider_val, self.binary_slider_val )
            
            # save palette
            # 'ascontiguousarray' to make a C contiguous copy 
            self.palette = np.ascontiguousarray( np.clip( 0, 255, palette2swatch( palette ) * 255. ).astype( np.uint8 ).transpose( ( 1, 0, 2 ) ) )
            
            # convert to uint8 format
            self.posterized_image_wo_smooth = np.clip( 0, 255, post_img*255. ).astype( np.uint8 )
            
            # post-smoothing
            self.posterized_image_w_smooth = post_smoothing( PIL.Image.fromarray( self.posterized_image_wo_smooth, 'RGB' ), self.blur_slider_val, blur_window = self.blur_window_slider_val )
            
            end = time.time()
            print( "Finished. Total time: ", end - start )
            
            self.add_to_paletteList( self.palette )
            self.add_to_imageList( self.posterized_image_w_smooth )
            
            self.set_image( self.imageLabel, self.imageList[-1] )
            
            # update current index position
            self.current_image_indx = len( self.imageList ) - 1
    
    
    # re-smooth the image
    def smooth( self ):
        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first!' )
        else:
            if self.posterized_image_wo_smooth[0][0][0] == -1:
                QMessageBox.warning( self, 'Warning', 'Please posterize your image first' )
            else:
                print( "Start smoothing." )
                
                messagebox = TimerMessageBox( 1, self )
                messagebox.open()
                
                self.posterized_image_w_smooth = post_smoothing( PIL.Image.fromarray( self.posterized_image_wo_smooth, 'RGB' ), self.blur_slider_val, blur_window = self.blur_window_slider_val )
                print( "Smoothing Finished." )
                
                self.add_to_paletteList( self.paletteList[-1] )
                self.add_to_imageList( self.posterized_image_w_smooth )
                
                self.set_image( self.imageLabel, self.imageList[-1] )
                
                # update current index position
                self.current_image_indx = len( self.imageList ) - 1
                
                
    # function to save current image
    def save_current_image( self ):
        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first.' )
        else:
            if self.posterized_image_wo_smooth[0][0][0] == -1:
                QMessageBox.warning( self, 'Warning', 'Please posterize your image first.' )
            else:
                reply = QMessageBox.question( self, 'Message', "Are you sure to save your current image on this panel?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
                
                if reply == QMessageBox.Yes:
                    image_name = QFileDialog.getSaveFileName( self, 'Save Image' )
                    if not image_name:
                        return
                    Image.fromarray( self.imageList[self.current_image_indx] ).save( image_name[0] + '.png')
                else:
                    pass
    
    # function to save current image
    def save_current_palette( self ):
        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first.' )
        else:
            if self.posterized_image_wo_smooth[0][0][0] == -1:
                QMessageBox.warning( self, 'Warning', 'Please posterize your image first.' )
            else:
                reply = QMessageBox.question( self, 'Message', "Are you sure to save your current palette on this panel?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
                
                if reply == QMessageBox.Yes:
                    image_name = QFileDialog.getSaveFileName( self, 'Save Palette' )
                    if not image_name:
                        return
                    Image.fromarray( self.paletteList[self.current_image_indx] ).save( image_name[0] + '.png')
                else:
                    pass
    
    # load user's own blurring map
    def pop_up_load_saliency_map( self ):
        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first.' )
        else:
            if self.posterized_image_wo_smooth[0][0][0] == -1:
                QMessageBox.warning( self, 'Warning', 'Please posterize your image first.' )
            else:
                reply = QMessageBox.question( self, 'Message', "Do you have your own blurring map (in grayscale and in .jpg/.png extension)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
                
                if reply == QMessageBox.Yes:
                    map = QFileDialog.getOpenFileName( self, 'Select file' )
                    map_path = map[0]
                    
                    if map_path[-4:] not in ['.jpg', '.png']:
                        QMessageBox.warning( self, 'Warning', 'Please upload your map with .jpg or .png extension.' )
                        return
                        
                    self.saliency_map = cv2.imread( map[0] ) / 255
                    h_s, w_s, dim_s = self.saliency_map.shape
                    h_i, w_i, dim_i = self.input_image.shape
                    
                    if ( h_i, w_i ) != ( h_s, w_s ):
                        QMessageBox.warning( self, 'Warning', 'Please upload your map with size:\n\n ' + '    ' + str( h_i ) + ' x  ' + str( w_i ) + '\n\n' + 'You upload the map with size:\n\n ' + '    ' + str( h_s ) + ' x  ' + str( w_s ) )
                        return
                    
                    if not np.array_equal( self.saliency_map[:,:,0], self.saliency_map[:,:,1] ) or not np.array_equal( self.saliency_map[:,:,1], self.saliency_map[:,:,2] ):
                        QMessageBox.warning( self, 'Warning', 'Please upload your map with grayscale.' )
                        return
                    
                    print( "Start smoothing." )
                    
                    messagebox = TimerMessageBox( 1, self )
                    messagebox.open()
                    
                    self.posterized_image_w_smooth = post_smoothing(
                        PIL.Image.fromarray( self.posterized_image_wo_smooth, 'RGB' ),
                        self.blur_slider_val,
                        blur_window = self.blur_window_slider_val,
                        blur_map = self.saliency_map[:, :, 0]
                        )    # bugs: 'tuple' object is not callable
                    print( "Smoothing Finished." )
                    
                    self.add_to_paletteList( self.paletteList[-1] )
                    self.add_to_imageList( self.posterized_image_w_smooth )
                    
                    self.set_image( self.imageLabel, self.imageList[-1] )
                    
                    # update current index position
                    self.current_image_indx = len( self.imageList ) - 1
                    
    
    # Function if users tend to close the app
    def closeEvent( self, event ):
        reply = QMessageBox.question( self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication( sys.argv )
    ex = MainWindow()
    sys.exit( app.exec_() )
    
    
if __name__ == '__main__':
    main()
    