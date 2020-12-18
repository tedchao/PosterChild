#!/usr/bin/env python3

import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
#from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton


class MainWindow( QWidget ):
    
    def __init__( self ):
        super().__init__()
        
        self.title = 'Posterization'
        self.x = 300
        self.y = 300
        self.width = 100
        self.height = 100
        
        self.initUI()
    
    def initUI( self ):
        self.setWindowTitle( self.title )
        #self.setStyleSheet("background-color: white;") 
        
        # Set the welcome icon in GIF
        self.gif = QMovie( "simpson.gif" )
        self.imageLabel = QLabel()
        self.imageLabel.setMovie( self.gif )
        self.gif.start()
        
        # Variables
        self.imagePath = ""
        self.input_image = None
        self.posterized_image_wo_smooth = None
        self.posterized_image_w_smooth = None
        
        btns_box = QVBoxLayout() # set bottons' box
        img_box = QHBoxLayout() # set image's box
        
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
        
        
        
        btns_box.addStretch(1)
        btns_box.addWidget( self.img_btn )
        btns_box.addWidget( self.posterize_btn )
        btns_box.addWidget( self.smooth_btn )
        btns_box.addStretch(1)
        
        
        img_box.addWidget( self.imageLabel )
        
        #Set grid layout
        grid = QGridLayout()
        grid.addLayout( btns_box, 0, 0 )
        grid.addLayout( img_box, 0, 1 )
        self.setLayout(grid) 
        
        self.show()
    
    
    # Function for selecting an input image
    def get_image( self ):
        img = QFileDialog.getOpenFileName( self, 'Select file' )
        if img:
            path = img[0]
            self.load_image( path )
        else:
            self.fileLabel.setText( "No file selected" )
    
    
    #Load new image function
    def set_image( self, image ):
        #Load the image into the label
        self.imageLabel.setPixmap( image )
        
        
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
            pass
    
    
    # re-smooth the image
    def smooth( self ):
        print( "Start smoothing." )
        if self.imagePath == "":
            QMessageBox.warning( self,'Warning','Please select an image first!' )
        else:
            if self.posterized_image_wo_smooth == None:
                QMessageBox.warning( self,'Warning','Please posterize your image first!' )
                
    
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
    