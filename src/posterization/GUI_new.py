#!/usr/bin/env python3

import sys


from pathlib import Path
from .posterization_gui import *
from .simplepalettes import *
'''
from posterization_gui import *
import simplepalettes
'''

import numpy as np
import cv2

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
        self.welcome_img_path =  str( Path( __file__ ).parent / "car.jpg" )
        #self.welcome_img_path = "car.jpg"
        self.welcome = QPixmap( self.welcome_img_path )
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap( self.welcome )
        
        ### palette images
        self.paletteLabel = QLabel()
        self.paletteLabel.setPixmap( QPixmap() )
        
        #### Variables
        self.imagePath = ""
        self.palette = None
        
        # used for recoloring
        self.weights_per_pixel_smooth = None
        self.weights_per_pixel = None
        self.palette_recolor = None
        self.palette_og = None
        
        self.waitingtime = 1
        
        self.show_palette = 1
        self.show_input = 0
        
        self.live_smoothing = True
        
        self.blur_window_slider_val = 7 # default
        self.blur_slider_val = 0.1      # default
        self.binary_slider_val = 0.8    # default
        self.cluster_slider_val = 20    # default
        self.palette_slider_val = 6     # default
        self.blend_slider_val = 3       # default
        self.current_image_indx = -1    # Track the current image index in the image list
        
        self.imageList = []
        self.add_to_imageList( cv2.cvtColor( cv2.imread( self.welcome_img_path ), cv2.COLOR_BGR2RGB ) )
        
        self.paletteList = [-1 * np.ones( ( 1, 1, 1 ) )]
        
        self.input_image = None         # Store it as np array
        self.saliency_map = None
        self.posterized_image_wo_smooth = -1 * np.ones( ( 1, 1, 1 ) )      # Store it as np array
        self.posterized_image_w_smooth = -1 * np.ones( ( 1, 1, 1 ) )       # Store it as np array
        
        #### BOXES
        btns_io_box = QHBoxLayout() # set bottons' box for I/O
        
        # algorithm_btns_box = QVBoxLayout() # set bottons' box for algorithms
        sld_box_palette = QHBoxLayout()
        sld_box_blend = QHBoxLayout()
        sld_box_cluster = QHBoxLayout()
        sld_box_binary = QHBoxLayout()
        sld_box_blur = QHBoxLayout()
        sld_box_window = QHBoxLayout()
        
        btns_posterize_box = QHBoxLayout() # set bottons' box for posterization and reset
        
        sld_r_recolor = QHBoxLayout()
        sld_g_recolor = QHBoxLayout()
        sld_b_recolor = QHBoxLayout()
        
        blur_box = QHBoxLayout()
        
        img_box = QHBoxLayout() # set image's box
        pages_box = QVBoxLayout() # set next-previous box
        show_hide_box = QVBoxLayout()
        
        combo_recolor_box = QHBoxLayout()
        recolor_btn_box = QHBoxLayout()
        
        
        #### BUTTONS
        # button for selecting an input image
        self.img_btn = QPushButton( 'Choose Image' )
        self.img_btn.clicked.connect( self.get_image )
        self.img_btn.setToolTip( 'Press the button to <b>select</b> an image.' ) 
        self.img_btn.setMaximumWidth( 150 )
        
        # button for posterizing the given image
        self.posterize_btn = QPushButton( 'Posterize' )
        self.posterize_btn.clicked.connect( self.posterize )
        self.posterize_btn.setToolTip( 'Press the button to <b>posterize</b> your image.' ) 
        self.posterize_btn.setMaximumWidth( 110 )
        
        # button for reseting posterization parameters
        self.reset_posterize_btn = QPushButton( 'Reset' )
        self.reset_posterize_btn.clicked.connect( self.reset_posterize )
        self.reset_posterize_btn.setToolTip( 'Press the button to <b>reset</b> all posterization parameters.' ) 
        self.reset_posterize_btn.setMaximumWidth( 110 )
        
        # button for re-smoothing the posterized image
        self.smooth_btn = QPushButton( 'Re-Smooth' )
        self.smooth_btn.clicked.connect( self.smooth )
        self.smooth_btn.setToolTip( 'Press the button to <b>re-smooth</b> your posterized image.' ) 
        self.smooth_btn.setMaximumWidth( 150 )
        
        # button for loading the saliency map
        self.map_btn = QPushButton( 'Smooth with Custom Map' )
        self.map_btn.clicked.connect( self.pop_up_load_saliency_map )
        self.map_btn.setToolTip( 'Press the button to <b>load</b> your own map to blur.' ) 
        self.map_btn.setMaximumWidth( 180 )

        # button for saving the posterized image
        self.save_btn = QPushButton( 'Save Current Image' )
        self.save_btn.clicked.connect( self.save_current_image )
        self.save_btn.setToolTip( 'Press the button to <b>save</b> your current image.' ) 
        self.save_btn.setMaximumWidth( 150 )
        
        # button for saving the palette
        self.save_palette_btn = QPushButton( 'Save Palette' )
        self.save_palette_btn.clicked.connect( self.save_current_palette )
        self.save_palette_btn.setToolTip( 'Press the button to <b>save</b> the palette.' ) 
        self.save_palette_btn.setMaximumWidth( 150 )
        
        
        #### Previous-next buttons
        self.previous_btn = QPushButton( 'Previous Posterization' )
        self.previous_btn.clicked.connect( self.paste_previous_image )
        self.previous_btn.setToolTip( 'Press the button to see your <b>previous</b> image in the gallory.' ) 
        self.previous_btn.setMinimumWidth( 165 )
        self.previous_btn.setMaximumWidth( 165 )
        
        self.next_btn = QPushButton( 'Next Posterization' )
        self.next_btn.clicked.connect( self.paste_next_image )
        self.next_btn.setToolTip( 'Press the button to see your <b>next</b> image in the gallory.' ) 
        self.next_btn.setMinimumWidth( 165 )
        self.next_btn.setMaximumWidth( 165 )
        
        #### Show/Hide buttons
        self.palette_btn = QPushButton( 'Show/Hide Palette' )
        self.palette_btn.clicked.connect( self.show_hide_palette )
        self.palette_btn.setToolTip( 'Press the button to <b>show</b> or <b>hide</b> the palette.' ) 
        self.palette_btn.setMinimumWidth( 165 )
        self.palette_btn.setMaximumWidth( 165 )
        
        self.og_img_btn = QPushButton( 'Show/Hide Input Image' )
        self.og_img_btn.clicked.connect( self.show_hide_input_image )
        self.og_img_btn.setToolTip( 'Press the button to <b>show</b> your input image.' ) 
        self.og_img_btn.setMinimumWidth( 165 )
        self.og_img_btn.setMaximumWidth( 165 )
        
        
        #### SLIDERS
        # slider for palette size
        self.blend_sld = QSlider( Qt.Horizontal )
        self.blend_sld.setRange( 0, 15 )
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
        self.blur_text = QLabel( 'Detail abstraction (Default: 0.1):  ' )
        self.binary_text = QLabel( 'Region clumpiness (Default: 0.8):       ' )
        self.cluster_text = QLabel( 'Rare color suppression (Default: 20):' )
        self.palette_text = QLabel( 'Palette size (Default: 6):                     ' )
        self.blend_text = QLabel( 'Palette blends (Default: 3):                 ' )
        
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
        self.binary_sld_label = QLabel( '0.8' )
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
        
        
        ####
        ### combo boxes for recoloring
        ####
        self.combobox = QComboBox(self)
        self.combobox.setMaximumWidth(100)
        self.combotext = QLabel( 'Choose color: ' )
        
        self.r_slider_val = 0
        self.g_slider_val = 0
        self.b_slider_val = 0
        
        self.rgb_text = QLabel( 'Recolor the image via its palette:' )
        
        self.rgb_text.setMaximumWidth( 250 )
        
        self.r_sld_label = QLabel( '0' )
        self.r_sld_label.setAlignment( Qt.AlignLeft )
        self.r_sld_label.setMinimumWidth( 80 )
        self.r_sld_text_label = QLabel( 'R:' )
        self.r_sld_text_label.setAlignment( Qt.AlignLeft )
        
        self.g_sld_label = QLabel( '0' )
        self.g_sld_label.setAlignment( Qt.AlignLeft )
        self.g_sld_label.setMinimumWidth( 80 )
        self.g_sld_text_label = QLabel( 'G:' )
        self.g_sld_text_label.setAlignment( Qt.AlignRight )
        
        self.b_sld_label = QLabel( '0' )
        self.b_sld_label.setAlignment( Qt.AlignLeft )
        self.b_sld_label.setMinimumWidth( 80 )
        self.b_sld_text_label = QLabel( 'B:' )
        self.b_sld_text_label.setAlignment( Qt.AlignRight )
        
        # slider for palette recoloring
        self.r_sld = QSlider( Qt.Horizontal )
        self.r_sld.setRange( 0, 255 )
        self.r_sld.setFocusPolicy( Qt.NoFocus )
        self.r_sld.setSliderPosition( self.r_slider_val )
        self.r_sld.setPageStep( 1 )
        self.r_sld.setToolTip( 'Fine-tune the slider to get your desired recoloring for R-channel.' ) 
        self.r_sld.setMinimumWidth( 150 )
        self.r_sld.setMaximumWidth( 200 )
        self.r_sld.valueChanged.connect( self.r_change_slider )
        
        self.g_sld = QSlider( Qt.Horizontal )
        self.g_sld.setRange( 0, 255 )
        self.g_sld.setFocusPolicy( Qt.NoFocus )
        self.g_sld.setSliderPosition( self.g_slider_val )
        self.g_sld.setPageStep( 1 )
        self.g_sld.setToolTip( 'Fine-tune the slider to get your desired recoloring for G-channel.' ) 
        self.g_sld.setMinimumWidth( 150 )
        self.g_sld.setMaximumWidth( 200 )
        self.g_sld.valueChanged.connect( self.g_change_slider )
        
        self.b_sld = QSlider( Qt.Horizontal )
        self.b_sld.setRange( 0, 255 )
        self.b_sld.setFocusPolicy( Qt.NoFocus )
        self.b_sld.setSliderPosition( self.b_slider_val )
        self.b_sld.setPageStep( 1 )
        self.b_sld.setToolTip( 'Fine-tune the slider to get your desired recoloring for B-channel.' ) 
        self.b_sld.setMinimumWidth( 150 )
        self.b_sld.setMaximumWidth( 200 )
        self.b_sld.valueChanged.connect( self.b_change_slider )
        
        self.recolor_btn = QPushButton( 'Reset Current Color' )
        self.recolor_btn.clicked.connect( self.reset_current_recoloring )
        self.recolor_btn.setToolTip( 'Press the button to <b>reset</b> the current palette color.' ) 
        self.recolor_btn.setMinimumWidth( 150 )
        self.recolor_btn.setMaximumWidth( 150 )
        
        self.undo_recolor_btn = QPushButton( 'Reset All Colors' )
        self.undo_recolor_btn.clicked.connect( self.reset_all_recoloring )
        self.undo_recolor_btn.setToolTip( 'Press the button to <b>undo</b> your all previous recolorings.' ) 
        self.undo_recolor_btn.setMinimumWidth( 150 )
        self.undo_recolor_btn.setMaximumWidth( 150 )
        
        
        ### BOX FRAMES
        btns_io_box.addWidget( self.img_btn )
        btns_io_box.addWidget( self.save_btn )
        btns_io_box.addWidget( self.save_palette_btn )
        btns_io_box.addStretch(40)
        
        
        # Separate boxes for parameters
        sld_box_palette.addWidget( self.palette_text )
        sld_box_palette.addWidget( self.palette_sld )
        sld_box_palette.addWidget( self.palette_sld_label )
        sld_box_palette.addStretch(8)
        
        sld_box_blend.addWidget( self.blend_text )
        sld_box_blend.addWidget( self.blend_sld )
        sld_box_blend.addWidget( self.blend_sld_label )
        sld_box_blend.addStretch(8)
        
        sld_box_cluster.addWidget( self.cluster_text )
        sld_box_cluster.addWidget( self.cluster_sld )
        sld_box_cluster.addWidget( self.cluster_sld_label )
        sld_box_cluster.addStretch(8)
        
        sld_box_binary.addWidget( self.binary_text )
        sld_box_binary.addWidget( self.binary_sld )
        sld_box_binary.addWidget( self.binary_sld_label )
        sld_box_binary.addStretch(8)
        
        btns_posterize_box.addWidget( self.posterize_btn )
        btns_posterize_box.addWidget( self.reset_posterize_btn )
        btns_posterize_box.addStretch(8)
        
        sld_box_blur.addWidget( self.blur_text )
        sld_box_blur.addWidget( self.blur_sld )
        sld_box_blur.addWidget( self.blur_sld_label )
        sld_box_blur.addStretch(8)
        
        sld_box_window.addWidget( self.blur_window_text )
        sld_box_window.addWidget( self.blur_window_sld )
        sld_box_window.addWidget( self.blur_window_sld_label )
        sld_box_window.addStretch(8)
        
        # blur box for re-smooth and smooth by map
        blur_box.addWidget( self.smooth_btn )
        blur_box.addWidget( self.map_btn )
        blur_box.addStretch(8)
        
        # recoloring box
        combo_recolor_box.addWidget( self.combotext )
        combo_recolor_box.addWidget( self.combobox )
        combo_recolor_box.addStretch(8)
        
        sld_r_recolor.addWidget( self.r_sld_text_label )
        sld_r_recolor.addWidget( self.r_sld )
        sld_r_recolor.addWidget( self.r_sld_label )
        sld_r_recolor.addStretch(8)
        
        sld_g_recolor.addWidget( self.g_sld_text_label )
        sld_g_recolor.addWidget( self.g_sld )
        sld_g_recolor.addWidget( self.g_sld_label )
        sld_g_recolor.addStretch(8)
        
        sld_b_recolor.addWidget( self.b_sld_text_label )
        sld_b_recolor.addWidget( self.b_sld )
        sld_b_recolor.addWidget( self.b_sld_label )
        sld_b_recolor.addStretch(8)
        
        recolor_btn_box.addWidget( self.recolor_btn )
        recolor_btn_box.addWidget( self.undo_recolor_btn )
        recolor_btn_box.addStretch(8)
        
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
        grid.setSpacing(12)
        
        grid.addLayout( btns_io_box, 0, 0 )
        
        ### parameters for posterization
        grid.addLayout( sld_box_palette, 1, 0 )
        grid.addLayout( sld_box_blend, 2, 0 )
        grid.addLayout( sld_box_cluster, 3, 0 )
        grid.addLayout( sld_box_binary, 4, 0 )
        grid.addLayout( btns_posterize_box, 5, 0 )
        
        ### parameters for smoothing
        grid.addLayout( sld_box_blur, 7, 0 )
        grid.addLayout( sld_box_window, 8, 0 )
        grid.addLayout( blur_box, 9, 0 )
    
        ### boxes for previous/next and show/hide
        grid.addLayout( pages_box, 0, 10 )
        grid.addLayout( show_hide_box, 0, 11 )
        
        ### sliders for recoloring
        grid.addWidget( self.rgb_text, 11, 0 )
        grid.addLayout( combo_recolor_box, 12, 0 )
        grid.addLayout( sld_r_recolor, 13, 0 )
        grid.addLayout( sld_g_recolor, 14, 0 )
        grid.addLayout( sld_b_recolor, 15, 0 )
        grid.addLayout( recolor_btn_box, 16, 0 )
        
        grid.addLayout( img_box, 1, 1, 18, 18 )
        self.setLayout(grid)
        
        self.show()
    
    ### Recoloring functions
    def set_rgb_slider( self, color ):
        color = color * 255.
        self.r_change_slider( int( color[0] ) )
        self.g_change_slider( int( color[1] ) )
        self.b_change_slider( int( color[2] ) )
        
        self.r_sld.setSliderPosition( int( color[0] ) )
        self.g_sld.setSliderPosition( int( color[1] ) )
        self.b_sld.setSliderPosition( int( color[2] ) )
        
    def onActivated( self, text ):
        color_indx = int( text ) - 1
        color = self.palette_recolor[ color_indx ]
        self.set_rgb_slider( color )
    
    def set_combo_icon( self ):
        self.combobox.clear()     # reset combo box
        
        for i in range( len( self.palette_recolor ) ):
            self.combobox.addItem( str( i + 1 ) )
            
        self.combobox.activated[str].connect( self.onActivated )
        default_color = self.palette_recolor[0]
        self.set_rgb_slider( default_color )
    
    def get_recolor_img_and_palette( self ):
        #recolor_img = ( self.weights_per_pixel @ self.palette_recolor ).reshape( self.input_image.shape )
        recolor_smooth_img = np.clip( 0, 255, ( self.weights_per_pixel_smooth @ self.palette_recolor ).reshape( self.input_image.shape ) * 255. ).astype( np.uint8 )
        #recolor_smooth_img = post_smoothing( PIL.Image.fromarray( np.clip( 0, 255, recolor_img * 255. ).astype( np.uint8 ), 'RGB' ),
            #self.blur_slider_val, blur_window = self.blur_window_slider_val )
        
        new_palette = np.ascontiguousarray( np.clip( 0, 255, simplepalettes.palette2swatch( self.palette_recolor ) * 
            255. ).astype( np.uint8 ).transpose( ( 1, 0, 2 ) ) )
        
        return recolor_smooth_img, new_palette
    
    def recolor_via_palette( self ):
        color_indx = int( self.combobox.currentText() ) - 1
        r_value = self.r_sld.value()
        g_value = self.g_sld.value()
        b_value = self.b_sld.value()
        
        self.palette_recolor[ color_indx ] = np.array([ r_value, g_value, b_value ]) / 255.
        recolor_img, new_palette = self.get_recolor_img_and_palette()
            
        self.add_to_paletteList( new_palette )
        self.add_to_imageList( recolor_img )
            
        self.set_image( self.imageLabel, self.imageList[-1] )
        self.set_image( self.paletteLabel, self.paletteList[-1] )
            
        # update current index position
        self.current_image_indx = len( self.imageList ) - 1
    
    def reset_current_recoloring( self ):
        if self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'Please posterize your image first' )
        else:
            # visualization for current combox text
            color_indx = int( self.combobox.currentText() ) - 1
            current_color = self.palette_og[ color_indx ]
            self.set_rgb_slider( current_color )
            
            self.palette_recolor[ color_indx ] = self.palette_og[ color_indx ].copy()
            
            recolor_img, new_palette = self.get_recolor_img_and_palette()
            
            self.add_to_paletteList( new_palette )
            self.add_to_imageList( recolor_img )
            
            self.set_image( self.imageLabel, self.imageList[-1] )
            self.set_image( self.paletteLabel, self.paletteList[-1] )
            
            # update current index position
            self.current_image_indx = len( self.imageList ) - 1
    
    def reset_all_recoloring( self ):
        if self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'Please posterize your image first' )
        else:
            # visualization for current combox text
            color_indx = int( self.combobox.currentText() ) - 1
            current_color = self.palette_og[ color_indx ]
            self.set_rgb_slider( current_color )
            
            self.palette_recolor = self.palette_og.copy()
            
            self.add_to_paletteList( self.palette )
            self.add_to_imageList( self.posterized_image_w_smooth )
            
            self.set_image( self.imageLabel, self.imageList[-1] )
            self.set_image( self.paletteLabel, self.paletteList[-1] )
            
            # update current index position
            self.current_image_indx = len( self.imageList ) - 1
            
    ### Reset for posterization parameters
    def reset_posterize( self ):
        self.binary_change_slider( 80 )
        self.cluster_change_slider( 20 )
        self.palette_change_slider( 6 )
        self.blend_change_slider( 3 )
        
        self.binary_sld.setSliderPosition( 80 )
        self.cluster_sld.setSliderPosition( 20 )
        self.palette_sld.setSliderPosition( 6 )
        self.blend_sld.setSliderPosition( 3 )
        
        self.binary_sld.repaint()
        self.cluster_sld.repaint()
        self.palette_sld.repaint()
        self.blend_sld.repaint()
    
    
    ### Slider functions
    def r_change_slider(self, value):
        self.r_slider_val = value
        self.r_sld_label.setText( str( value ) )
        if self.live_smoothing: self.recolor_via_palette()
    
    def g_change_slider(self, value):
        self.g_slider_val = value
        self.g_sld_label.setText( str( value ) )
        if self.live_smoothing: self.recolor_via_palette()
    
    def b_change_slider(self, value):
        self.b_slider_val = value
        self.b_sld_label.setText( str( value ) )
        if self.live_smoothing: self.recolor_via_palette()
    
    def blur_window_change_slider(self, value):
        self.blur_window_slider_val = 2 * value + 3
        self.blur_window_sld_label.setText( str( 2 * value + 3 ) )
        if self.live_smoothing: self.smooth()
        
    def blur_change_slider(self, value):
        self.blur_slider_val = value / 100
        self.blur_sld_label.setText( str( value / 100 ) )
        if self.live_smoothing: self.smooth()
    
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
                self.paletteLabel.repaint()
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
                self.paletteLabel.repaint()
            self.set_image( self.imageLabel, self.imageList[self.current_image_indx] )
    
    #Load new image function
    def set_image( self, panel, image ):
        #Load the image into the label
        height, width, dim = image.shape
        qim = QImage( image.data, width, height, 3 * width, QImage.Format_RGB888 )
        panel.setPixmap( QPixmap( qim ) )
        panel.repaint()
    
    def add_to_imageList( self, image ):
        self.imageList.append( np.asarray( image ) )
        
    def add_to_paletteList( self, palette ):
        self.paletteList.append( np.asarray( palette ) )
        
    def load_image( self, path ):
        print ( "Loading Image." )
        
        self.imageList = []     # initialized back to empty when giving another input image
        self.paletteList = [-1 * np.ones( ( 1, 1, 1 ) )]
        self.paletteLabel.setPixmap( QPixmap() )
        
        
        # push input image in the list
        self.current_image_indx += 1
        self.input_image = cv2.cvtColor( cv2.imread( path ), cv2.COLOR_BGR2RGB )
        self.add_to_imageList( self.input_image )
        
        self.imageLabel.setPixmap( QPixmap( path ) )
        self.imagePath = path
    
    def show_hide_palette( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
            
        if self.paletteList[-1][0, 0, 0] == -1:
            QMessageBox.warning( self, 'Warning', 'You do not have palette. Please posterize the image first.' )
        else:
            self.show_palette = 1 - self.show_palette
            if self.current_image_indx != 0 and self.show_palette == 1:
                self.set_image( self.paletteLabel, self.paletteList[self.current_image_indx] )
            else:   # input image has no palette, so place a blank
                self.paletteLabel.setPixmap( QPixmap() )
        
    def show_hide_input_image( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
        if self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'This is your input image.' )
        else:
            self.show_input = 1 - self.show_input
            if self.show_input == 1:
                self.set_image( self.imageLabel, self.imageList[0] )
            else:
                self.set_image( self.imageLabel, self.imageList[self.current_image_indx] )
    
    # posterization
    def posterize( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
        #else:
        
        if self.imagePath == "":
            img_arr = np.asfarray( PIL.Image.open( self.welcome_img_path ).convert( 'RGB' ) ) / 255.
            self.input_image = img_arr
            path = self.welcome_img_path
        else:
            img_arr = np.asfarray( PIL.Image.open( self.imagePath ).convert( 'RGB' ) ) / 255.
            path = self.imagePath
        
        width, height, dim = img_arr.shape
        length = max( width, height )
        
        self.message = "This image has size " + str( height ) + ' x ' + str( width ) + '.\n\n'
        
        if length >= 1800:
            self.message += 'This is a large image and may take more than 8 mins to process.\n' + 'We suggest you posterize a downsized version to select appropriate parameters or vectorize the output.\n\n'
        else:
            if 500 < length < 600:
                self.waitingtime = 2
            elif 600 < length < 1000:
                self.waitingtime = 3
            elif 1000 <= length:
                self.waitingtime = 4
            self.message += 'This will take roughly ' + str( self.waitingtime ) + ' minutes to process.\n\n'
            
        reply = QMessageBox.question( self, 'Message', self.message + 'Do you want to proceed and posterize the image?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        
        if reply == QMessageBox.Yes:
            print( "Start posterizing." )
            
            # algorithm starts
            start = time.time()
                
            messagebox = TimerMessageBox( 1, self )
            messagebox.open()
            
            # K-means
            img_arr_re = img_arr.reshape( ( -1, 3 ) )
            img_arr_cluster = get_kmeans_cluster_image( self.cluster_slider_val, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
            
            # MLO
            post_img, final_colors, add_mix_layers, palette = \
            posterization( path, img_arr, img_arr_cluster, self.palette_slider_val, self.blend_slider_val, self.binary_slider_val )
            
            self.weights_per_pixel = add_mix_layers # save weight list per pixel
            
            # save palette
            # 'ascontiguousarray' to make a C contiguous copy 
            self.palette = np.ascontiguousarray( np.clip( 0, 255, simplepalettes.palette2swatch( palette ) * 255. ).astype( np.uint8 ).transpose( ( 1, 0, 2 ) ) )
                
            # convert to uint8 format
            self.posterized_image_wo_smooth = np.clip( 0, 255, post_img * 255. ).astype( np.uint8 )
            
            # make a map from unique colors to weights
            unique_colors, unique_indices = np.unique( self.posterized_image_wo_smooth.reshape( -1, 3 ), return_index = True, axis = 0 )
            color2weights = {}
            for col, index in zip( unique_colors, unique_indices ):
                weights = self.weights_per_pixel[ index ]
                color2weights[ tuple( col ) ] = weights

            # post-smoothing
            self.posterized_image_w_smooth = post_smoothing( PIL.Image.fromarray( self.posterized_image_wo_smooth, 'RGB' ), self.blur_slider_val, blur_window = self.blur_window_slider_val )
            
            # pass smoothing along to the weights
            self.weights_per_pixel_smooth = self.weights_per_pixel.copy()
            for col, weights in color2weights.items():
                #color_mask = ( self.posterized_image_w_smooth.reshape( -1, 3 ) == np.array( col ) [None,:] ).all()
                color_mask = np.where( np.all( self.posterized_image_w_smooth.reshape( -1, 3 ) == np.array( col ), axis = 1 ) )[0]
                self.weights_per_pixel_smooth[ color_mask ] = weights
            self.weights_per_pixel_smooth.shape = self.weights_per_pixel.shape
            
            ### setting for recoloring
            self.palette_recolor = palette  # save for palette recoloring
            self.palette_og = self.palette_recolor.copy()
            self.set_combo_icon()
            
            end = time.time()
            print( "Finished. Total time: ", end - start )
                
            self.add_to_paletteList( self.palette )
            self.add_to_imageList( self.posterized_image_w_smooth )
                
            self.set_image( self.imageLabel, self.imageList[-1] )
            self.set_image( self.paletteLabel, self.paletteList[-1] )
                
            # update current index position
            self.current_image_indx = len( self.imageList ) - 1
            
        else:
            pass
    
    
    # re-smooth the image
    def smooth( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self,'Warning','Please select an image first!' )
        #else:
        if self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'Please posterize your image first' )
        else:
            print( "Start smoothing." )
                
            #messagebox = TimerMessageBox( 1, self )
            #messagebox.open()
                
            self.posterized_image_w_smooth = post_smoothing( PIL.Image.fromarray( self.posterized_image_wo_smooth, 'RGB' ), self.blur_slider_val, blur_window = self.blur_window_slider_val )
            print( "Smoothing Finished." )
                
            self.add_to_paletteList( self.paletteList[-1] )
            self.add_to_imageList( self.posterized_image_w_smooth )
                
            self.set_image( self.imageLabel, self.imageList[-1] )
                
            # update current index position
            self.current_image_indx = len( self.imageList ) - 1
                
                
    # function to save current image
    def save_current_image( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self,'Warning','Please select an image first.' )
        #else:
        if self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'Please posterize your image first.' )
        else:
            reply = QMessageBox.question( self, 'Message', "Are you sure to save your current image on this panel?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
                
            if reply == QMessageBox.Yes:
                image_name = QFileDialog.getSaveFileName( self, 'Save Image' )
                if not image_name:
                    return
                
                if image_name[0][-4:] in ['.jpg', '.png']:
                    path_name = image_name[0]
                else:
                    path_name = image_name[0] + '.png'
                    
                Image.fromarray( self.imageList[self.current_image_indx] ).save( path_name )
            else:
                pass
    
    # function to save current image
    def save_current_palette( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self,'Warning','Please select an image first.' )
        #else:
        if self.posterized_image_wo_smooth[0][0][0] == -1:
            QMessageBox.warning( self, 'Warning', 'Please posterize your image first.' )
        else:
            reply = QMessageBox.question( self, 'Message', "Are you sure to save your current palette on this panel?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
                
            if reply == QMessageBox.Yes:
                image_name = QFileDialog.getSaveFileName( self, 'Save Palette' )
                if not image_name:
                    return
                
                if image_name[0][-4:] in ['.jpg', '.png']:
                    path_name = image_name[0]
                else:
                    path_name = image_name[0] + '.png'
                    
                Image.fromarray( self.paletteList[self.current_image_indx] ).save( path_name )
            else:
                pass
    
    # load user's own blurring map
    def pop_up_load_saliency_map( self ):
        #if self.imagePath == "":
        #    QMessageBox.warning( self,'Warning','Please select an image first.' )
        #else:
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
                )  
                print( "Smoothing Finished." )
                    
                self.add_to_paletteList( self.paletteList[-1] )
                self.add_to_imageList( self.posterized_image_w_smooth )
                    
                self.set_image( self.imageLabel, self.imageList[-1] )
                    
                # update current index position
                self.current_image_indx = len( self.imageList ) - 1
                    
    
    # Function if users tend to close the app
    def closeEvent( self, event ):
        reply = QMessageBox.question( self, 'Message', "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        
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
    