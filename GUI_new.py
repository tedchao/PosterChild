#!/usr/bin/env python3

import sys
from posterization_gui import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# In command line: "pip install opencv-python-headless" to avoid qt complaining two set of binaries

#from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton


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
        self.gif = QMovie( "simpson.gif" )
        self.imageLabel = QLabel()
        self.imageLabel.setMovie( self.gif )
        self.gif.start()
        
        #### Variables
        self.imagePath = ""
        self.blur_slider_val = 0.1      # default
        self.binary_slider_val = 1.0    # default
        self.cluster_slider_val = 20    # default
        self.palette_slider_val = 6     # default
        self.blend_slider_val = 3       # default
        self.input_image = None         # Store it as PIL image
        self.current_image = None       # Store it as PIL image
        self.posterized_image_wo_smooth = None      # Store it as PIL image
        self.posterized_image_w_smooth = None       # Store it as PIL image
        
        
        #### BOXES
        btns_box = QVBoxLayout() # set bottons' box
        img_box = QHBoxLayout() # set image's box
        
        
        #### BUTTONS
        # button for selecting an input image
        self.img_btn = QPushButton( 'Select an image!' )
        self.img_btn.clicked.connect( self.get_image )
        self.img_btn.setToolTip( 'Press the button to <b>select</b> an image.' ) 
        self.img_btn.setMaximumWidth( 150 )
        
        # button for posterizing the given image
        self.posterize_btn = QPushButton( 'Posterize!' )
        self.posterize_btn.clicked.connect( self.posterize )
        self.posterize_btn.setToolTip( 'Press the button to <b>posterize</b> your image.' ) 
        self.posterize_btn.setMaximumWidth( 150 )
        
        # button for re-smoothing the posterized image
        self.smooth_btn = QPushButton( 'Re-smooth?' )
        self.smooth_btn.clicked.connect( self.smooth )
        self.smooth_btn.setToolTip( 'Press the button to <b>re-smooth</b> your posterized image.' ) 
        self.smooth_btn.setMaximumWidth( 150 )
        
        # button for re-smoothing the posterized image
        self.save_btn = QPushButton( 'Save current image' )
        self.save_btn.clicked.connect( self.save_current_image )
        self.save_btn.setToolTip( 'Press the button to <b>save</b> your current image.' ) 
        self.save_btn.setMaximumWidth( 150 )
        
        
        #### SLIDERS
        # slider for palette size
        self.blend_sld = QSlider( Qt.Horizontal )
        self.blend_sld.setRange( 1, 25 )
        self.blend_sld.setFocusPolicy( Qt.NoFocus )
        self.blend_sld.setSliderPosition( self.blend_slider_val )
        self.blend_sld.setPageStep( 1 )
        self.blend_sld.setToolTip( 'Fine-tune the slider to get your desired main palette size.' ) 
        self.blend_sld.setMaximumWidth( 200 )
        self.blend_sld.valueChanged.connect( self.blend_change_slider )
        
        # slider for palette size
        self.palette_sld = QSlider( Qt.Horizontal )
        self.palette_sld.setRange( 4, 15 )
        self.palette_sld.setFocusPolicy( Qt.NoFocus )
        self.palette_sld.setSliderPosition( self.palette_slider_val )
        self.palette_sld.setPageStep( 1 )
        self.palette_sld.setToolTip( 'Fine-tune the slider to get your desired main palette size.' ) 
        self.palette_sld.setMaximumWidth( 200 )
        self.palette_sld.valueChanged.connect( self.palette_change_slider )
        
        # slider for number of clusters for Kmeans
        self.cluster_sld = QSlider( Qt.Horizontal )
        self.cluster_sld.setRange( 15, 50 )
        self.cluster_sld.setFocusPolicy( Qt.NoFocus )
        self.cluster_sld.setSliderPosition( self.cluster_slider_val )
        self.cluster_sld.setPageStep( 1 )
        self.cluster_sld.setToolTip( 'Fine-tune the slider to get your desired threshold for outlier colors.' ) 
        self.cluster_sld.setMaximumWidth( 200 )
        self.cluster_sld.valueChanged.connect( self.cluster_change_slider )
        
        # slider for binary penalization
        self.binary_sld = QSlider( Qt.Horizontal )
        self.binary_sld.setRange( 0, 200 )
        self.binary_sld.setFocusPolicy( Qt.NoFocus )
        self.binary_sld.setSliderPosition( int( 100 * self.binary_slider_val ) )
        self.binary_sld.setPageStep( 1 )
        self.binary_sld.setToolTip( 'Fine-tune the slider to get your desired penalization on binary term.' ) 
        self.binary_sld.setMaximumWidth( 200 )
        self.binary_sld.valueChanged.connect( self.binary_change_slider )
        
        # slider for blurring threshold
        self.blur_sld = QSlider( Qt.Horizontal )
        self.blur_sld.setRange( 0, 100 )
        self.blur_sld.setFocusPolicy( Qt.NoFocus )
        self.blur_sld.setSliderPosition( int( 100 * self.blur_slider_val ) )
        self.blur_sld.setPageStep( 1 )
        self.blur_sld.setToolTip( 'Fine-tune the slider to get your desired blurring threshold.' ) 
        self.blur_sld.setMaximumWidth( 200 )
        self.blur_sld.valueChanged.connect( self.blur_change_slider )
        
        
        ### LABELS
        # labels
        self.blur_text = QLabel( 'Blurring Threshold (Default: 0.1):' )
        self.binary_text = QLabel( 'Richness value (Default: 1.0):' )
        self.cluster_text = QLabel( 'Outlier value (Default: 20):' )
        self.palette_text = QLabel( 'Main palette size (Default: 6):' )
        self.blend_text = QLabel( 'Number of blending ways (Default: 3):' )
        
        self.blur_text.setMaximumWidth( 230 )
        self.binary_text.setMaximumWidth( 230 )
        self.cluster_text.setMaximumWidth( 230 )
        self.palette_text.setMaximumWidth( 230 )
        self.blend_text.setMaximumWidth( 230 )
        
        # label text for blur slider
        self.blur_sld_label = QLabel( '0.1' )
        self.blur_sld_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.blur_sld_label.setMinimumWidth( 80 )
        
        # label text for binary penalization slider
        self.binary_sld_label = QLabel( '1.0' )
        self.binary_sld_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.binary_sld_label.setMinimumWidth( 80 )
        
        # label text for kmeans cluster slider
        self.cluster_sld_label = QLabel( '20' )
        self.cluster_sld_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.cluster_sld_label.setMinimumWidth( 80 )
        
        # label text for palette size slider
        self.palette_sld_label = QLabel( '6' )
        self.palette_sld_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.palette_sld_label.setMinimumWidth( 80 )
        
        # label text for blending way slider
        self.blend_sld_label = QLabel( '3' )
        self.blend_sld_label.setAlignment( Qt.AlignCenter | Qt.AlignVCenter )
        self.blend_sld_label.setMinimumWidth( 80 )
        
        
        ### BOX FRAMES
        btns_box.addStretch(1)
        
        btns_box.addWidget( self.img_btn )
        btns_box.addWidget( self.posterize_btn )
        btns_box.addWidget( self.smooth_btn )
        btns_box.addWidget( self.save_btn )
        
        btns_box.addStretch(20)
        btns_box.addStretch(2)
        btns_box.addWidget( self.palette_text )
        btns_box.addWidget( self.palette_sld )
        btns_box.addWidget( self.palette_sld_label )
        btns_box.addStretch(2)
        
        btns_box.addStretch(2)
        btns_box.addWidget( self.blend_text )
        btns_box.addWidget( self.blend_sld )
        btns_box.addWidget( self.blend_sld_label )
        btns_box.addStretch(2)
        
        btns_box.addStretch(2)
        btns_box.addWidget( self.cluster_text )
        btns_box.addWidget( self.cluster_sld )
        btns_box.addWidget( self.cluster_sld_label )
        btns_box.addStretch(2)
        
        btns_box.addStretch(2)
        btns_box.addWidget( self.binary_text )
        btns_box.addWidget( self.binary_sld )
        btns_box.addWidget( self.binary_sld_label )
        btns_box.addStretch(2)
        
        btns_box.addStretch(2)
        btns_box.addWidget( self.blur_text )
        btns_box.addWidget( self.blur_sld )
        btns_box.addWidget( self.blur_sld_label )
        btns_box.addStretch(2)
        btns_box.addStretch(20)
        
        btns_box.addStretch(1)
        
        
        img_box.addWidget( self.imageLabel )
        
        # Set grid layout
        grid = QGridLayout()
        grid.addLayout( btns_box, 0, 0 )
        grid.addLayout( img_box, 0, 1 )
        self.setLayout(grid) 
        
        self.show()
    
    # Slider functions
    def blur_change_slider(self, value):
        self.blur_slider_val = value
        self.blur_sld_label.setText( str( value / 100 ) )
    
    def binary_change_slider(self, value):
        self.binary_slider_val = value
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
            self.fileLabel.setText( "No file selected" )
    
    
    #Load new image function
    def set_image( self, panel, image ):
        #Load the image into the label
        panel.setPixmap( image )
        
        
    def load_image( self, path ):
        print ( "Loading Image." )
        self.imageLabel.setPixmap( QPixmap( path ) )
        self.imagePath = path
    
    
    # posterization
    def posterize( self ):
        print( "Start posterizing." )

        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first!' )
        else:
            img_arr = np.asfarray( PIL.Image.open( self.imagePath ).convert( 'RGB' ) ) / 255.
            self.input_image = img_arr
            
            # algorithm starts
            start = time.time()
            
            # K-means
            img_arr_re = img_arr.reshape( ( -1, 3 ) )
            img_arr_cluster = get_kmeans_cluster_image( self.cluster_slider_val, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
            
            # MLO
            post_img, final_colors, add_mix_layers, palette = \
            posterization( self.imagePath, self.input_image, img_arr_cluster, self.palette_slider_val, self.blend_slider_val, self.binary_slider_val )
            
            # convert to uint8 format
            self.posterized_image_wo_smooth = PIL.Image.fromarray( np.clip( 0, 255, post_img*255. ).astype( np.uint8 ), 'RGB' )
            
            # post-smoothing
            self.posterized_image_w_smooth = post_smoothing( self.posterized_image_wo_smooth, self.blur_slider_val )
            
            end = time.time()
            print( "Finished. Total time: ", end - start )
    
    
    # re-smooth the image
    def smooth( self ):
        print( "Start smoothing." )
        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first!' )
        else:
            if self.posterized_image_wo_smooth == None:
                QMessageBox.warning( self,'Warning','Please posterize your image first!' )
                
    # function to save current image
    def save_current_image( self ):
        pass
        

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
    