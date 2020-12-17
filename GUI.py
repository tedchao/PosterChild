#!/usr/bin/env python3
from posterization_gui import *
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox


def posterized_pipline( path, img_arr, img_og, threshold, num_clusters, num_blend, palette_num, penalization ):
    global tk_posterized_image, tk_palette_color, tk_add_mix, tk_img_shape, smooth_posterized_image
    
    # algorithm starts
    start = time.time()
    
    # K-means
    img_arr_re = img_arr.reshape( ( -1, 3 ) )
    img_arr_cluster = get_kmeans_cluster_image( num_clusters, img_arr_re, img_arr.shape[0], img_arr.shape[1] )
    
    # MLO
    post_img, final_colors, add_mix_layers, palette = \
    posterization( path, img_og, img_arr_cluster, palette_num, num_blend, penalization )
    tk_img_shape = post_img.shape
    
    # convert to uint8 format
    post_img = PIL.Image.fromarray( np.clip( 0, 255, post_img*255. ).astype( np.uint8 ), 'RGB' )
    tk_posterized_image = post_img   # this image is only posterized without smoothing
    tk_palette_color = palette
    tk_add_mix = add_mix_layers
    
    # post-smoothing
    post_img = post_smoothing( post_img, threshold )
    smooth_posterized_image = post_img
    
    end = time.time()
    print( "Finished. Total time: ", end - start )
    
    return post_img

    
def select_image():
    global panel, path, tk_input_image, tk_switch, tk_posterized_image
    global tk_num_clusters, tk_palette_size, tk_num_blend, tk_thres
    global tk_pal_num, tk_rc_r, tk_rc_g, tk_rc_b, tk_pc
    
    # assignment on global variables
    path = tkinter.filedialog.askopenfilename()
    tk_input_image = PIL.Image.open( path ).convert( 'RGB' )    # pillow image
    
    # convert Pillow image to ImageTK
    tk_image = PIL.ImageTk.PhotoImage( tk_input_image )
    
    if panel is None:
        panel = Label( image = tk_image )
        panel.image = tk_image
        panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
        
    else:
        tk_posterized_image = None
        panel.configure( image = tk_image )
        panel.image = tk_image
        
    c_pc = Label( root, text = 'Penalization on binary term (default: 1): ')
    #c_pc_format = Label( root, text = 'Format: R, G, B (0 to 255)')
    tk_pc = Entry(root)
    c_pc.grid(row=18, column=0)
    #c_pc_format.grid(row=18, column=0)
    tk_pc.grid(row=19, column=0)
    
    c_kms = Label( root, text = 'Threshold for outlier colors (default: 20): ')
    tk_num_clusters = Entry(root)
    c_kms.grid(row=20, column=0)
    tk_num_clusters.grid(row=21, column=0)
    
    p_sz = Label( root, text = 'Main palette size (default: 6): ')
    tk_palette_size = Entry(root)
    p_sz.grid(row=22, column=0)
    tk_palette_size.grid(row=23, column=0)
    
    n_b = Label( root, text = 'Numbers of blending ways (default: 3): ')
    tk_num_blend = Entry(root)
    n_b.grid(row=24, column=0)
    tk_num_blend.grid(row=25, column=0)
    
    thres = Label( root, text = 'Blurring threshold (0 to 1, default: 0.1): ')
    tk_thres = Entry(root)
    thres.grid(row=26, column=0)
    tk_thres.grid(row=27, column=0)
    
    
def posterize_button():
    global tk_posterized_image
    
    if panel is None:
        tkinter.messagebox.showwarning(title='Warning', message='Please select an image first.')
        
    else:
        # true image to work on
        img_arr = np.asfarray( PIL.Image.open( path ).convert( 'RGB' ) ) / 255. 
        img_arr_og = np.copy( img_arr )
        
        
        if tk_num_clusters.get():
            num_clusters = int( tk_num_clusters.get() )
        else:
            num_clusters = 20
            
        if tk_palette_size.get():
            palette_size = int( tk_palette_size.get() )
        else:
            palette_size = 6
            
        if tk_num_blend.get():
            num_blend = int( tk_num_blend.get() )
        else:
            num_blend = 3
            
        if tk_thres.get():
            threshold = float( tk_thres.get() )
        else:
            threshold = 0.1
            
        if tk_pc.get():
            penalization = float( tk_pc.get() )
        else:
            penalization = 1
        
        posterized_image = posterized_pipline( path, img_arr, img_arr_og, threshold, num_clusters, num_blend, palette_size, penalization )
        
        tk_switch = 1
        panel.image.paste( posterized_image )
        panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
        print( 'palette colors: ', np.clip( 0, 255, tk_palette_color * 255. ).astype( np.uint8 ) )
        
        
def smooth_image():
    global tk_posterized_image, smooth_posterized_image
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please posterize the image before smoothing the bouandries.' )
            
        else:
            if not tk_thres.get():
                tkinter.messagebox.showwarning( title='Warning', message='Please specify blurring threshold.' )
                
            else:
                smooth_posterized_image = post_smoothing( tk_posterized_image, float( tk_thres.get() ) )
                panel.image.paste( smooth_posterized_image )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
                
def compare():
    global tk_switch
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please posterize the image before comparing with the original image.' )
            
        elif smooth_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please smooth the posterized image before comparing with the original image.' )
            
        else:
            if tk_switch == 0:
                panel.image.paste( tk_input_image )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
                tk_switch = 1
            else:
                panel.image.paste( smooth_posterized_image )
                panel.grid(row = 0, column = 1, columnspan = 2, rowspan = 200)
                tk_switch = 0
                
                
def savefile():
    
    if panel is None:
        tkinter.messagebox.showwarning( title='Warning', message='Please select an image first.' )
        
    else:
        if tk_posterized_image is None:
            tkinter.messagebox.showwarning(title='Warning', message='Please posterize the image before saving it.')
            
        elif smooth_posterized_image is None:
            tkinter.messagebox.showwarning( title='Warning', message='Please smooth the posterized image before saving it.' )
            
        else:
            filename = tkinter.filedialog.asksaveasfilename( defaultextension=".png" )
            if not filename:
                return
            smooth_posterized_image.save( filename )
        
            
root = Tk()
root.title( 'Posterization' )

panel = None
tk_switch = 0
tk_posterized_image = None
smooth_posterized_image = None

f1 = Frame(root)
btn1 = Button(f1, text="Select an image", command = select_image).pack(side=TOP, fill="both", expand="yes")
btn2 = Button(f1, text="Posterize!", command = posterize_button).pack(side=TOP, fill="both", expand="yes")
btn3 = Button(f1, text="Re-smooth?", command = smooth_image).pack(side=TOP, fill="both", expand="yes")
btn4 = Button(f1, text="Press to compare", command = compare).pack(side=TOP, fill="both", expand="yes")
btn5 = Button(f1, text="Save posterized image", command = savefile).pack(side=TOP, fill="both", expand="yes")
f1.grid(row=0, column=0)

# kick off the GUI
root.mainloop()

    
    