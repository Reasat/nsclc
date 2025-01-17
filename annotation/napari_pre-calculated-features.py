#!/usr/bin/env python
# coding: utf-8

# In[4]:




# In[5]:


import numpy as np
import napari
import cv2
import matplotlib.pyplot as plt
from IPython.core.display import display
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import argparse
import os
import napari_lazy_openslide
import openslide

parser=argparse.ArgumentParser()
parser.add_argument('--filePath', default = 'E:/Data/dhsr/RD/S12-13045-1-A-9_12-2414.scn')
parser.add_argument('--featurePath', default = 'E:/Data/dhsr/features/S12-13045-1-A-9_12-2414_resnet50_tile-1024_level-1_pca-50.npz')
args = parser.parse_args()



# In[34]:


filename_wsi = os.path.basename(args.filePath).replace('.scn','')


# In[35]:


path = args.filePath
# path = '/home/reasat/data/dhsr/RD/S12-10712-1-B-2_12-2010.scn'
# path = '/home/reasat/data/dhsr/RD/S12-13562-1-A-6_12-2529.scn'


# In[36]:


wsi = openslide.OpenSlide(path)


# In[37]:


dict(wsi.properties)


# In[38]:


regions_dict = {key: wsi.properties[key] for key in wsi.properties.keys() if 'openslide.region' in key}
regions_dict


# In[12]:


# reform
regions_dict_rf = {}
for i in range(len(regions_dict)//4):
    regions_dict_rf['region-{}'.format(i)]={
        'x':int(regions_dict['openslide.region[{}].x'.format(i)]),
        'y':int(regions_dict['openslide.region[{}].y'.format(i)]),
        'width':int(regions_dict['openslide.region[{}].width'.format(i)]),
         'height': int(regions_dict['openslide.region[{}].height'.format(i)])
    }
        


img, prop = napari_lazy_openslide.lazy_openslide.reader_function(path)[0]
X_OFF = regions_dict_rf['region-0']['x']
Y_OFF = regions_dict_rf['region-0']['y']
tile_size = 1024

def create_square(x,y,tile_size):
    return [[y,x],
            [y+tile_size,x],
            [y+tile_size,x+tile_size],
            [y,x+tile_size]
           ]

def create_square(x, y, tile_size):
    return [[y, x],
            [y + tile_size, x],
            [y + tile_size, x + tile_size],
            [y, x + tile_size]
            ]


def has_tissue(wsi, x, y, tile_size, downsample_level=1):
    tile = wsi.read_region(
        location=(np.uint32(x), np.uint32(y)),
        level=downsample_level,
        size=(np.uint32(tile_size // wsi.level_downsamples[downsample_level]),
              np.uint32(tile_size // wsi.level_downsamples[downsample_level]))
    )
    tile = np.array(tile)[:, :, :3]
    mean = tile.reshape((-1, 3)).mean(axis=0)
    return ((mean < np.array([225] * 3)).all())

def make_int_rev(coord):
    x=np.uint32(X_OFF+(coord[1]-X_OFF)//tile_size*tile_size)
    y=np.uint32(Y_OFF+(coord[0]-Y_OFF)//tile_size*tile_size)
    return x,y

x_list = []
y_list = []
for key in regions_dict_rf.keys():
    region = regions_dict_rf[key]
    y = np.arange(region['y'],
                  region['y'] + region['height'] // tile_size * tile_size,
                  tile_size)
    x = np.arange(region['x'],
                  region['x'] + region['width'] // tile_size * tile_size,
                  tile_size)
    x_list.append(x)
    y_list.append(y)

yv, xv = np.meshgrid(np.concatenate(np.array(y_list)),
                     np.concatenate(np.array(x_list)))
xv = xv.flatten()
yv = yv.flatten()

grid_tissue, tissue_coords = zip(
    *[(create_square(x, y, tile_size), (x, y)) for x, y in zip(tqdm(xv), yv) if has_tissue(wsi, x, y, tile_size)])
print('No. of tiles with tissues', len(grid_tissue))

npz = np.load(args.featurePath)
features = npz['features_pca']

coord_feat_dict = {c: f for f, c in zip(features, tissue_coords)}

with napari.gui_qt():
    viewer = napari.view_image(img)
    layer_img = viewer.layers['img']
    points_layer_bg = viewer.add_points(
        name='points_bg',
        size = 100,
        face_color= 'blue'
    )
    shapes_layer_bg = viewer.add_shapes(name='tiles_bg',
                                        edge_color='blue',
                                       face_color='blue',
                                       opacity=0.3)
    
    @points_layer_bg.mouse_drag_callbacks.append
    def get_coord_bg(layer,event):
        if layer.mode == 'add':
            coord = make_int_rev(layer.coordinates)
            if coord not in coord_feat_dict.keys():
                # layer.remove_selected()
                print('Warning: Coord {} not found in feature list, maybe no tissue tile?'.format(coord))
            #         print(coord)
            # else:
            shape = create_square(
                coord[0], coord[1],
                tile_size
            )

            shapes_layer_bg.add(shape)
            # print('point num', len(layer.data))
            # print('shape num', len(shapes_layer_bg.data))
            # print('points', layer.data)
            # print('shapes', shapes_layer_bg.data)


    @points_layer_bg.bind_key('Backspace')
    @points_layer_bg.bind_key('Delete')
    def delete_selected(layer):
        """Delete all selected points."""
        if str( layer._mode) in ('select', 'add'):
            shapes_layer_bg.selected_data = points_layer_bg.selected_data
            shapes_layer_bg.remove_selected()
            layer.remove_selected()
    
    points_layer_fg = viewer.add_points(
        name='points_fg', 
        size=100,
        face_color='green'
    )
    shapes_layer_fg = viewer.add_shapes(
        name='tiles_fg',
        edge_color='green',
        face_color= 'green',
        opacity=0.3
    )
    
    @points_layer_fg.mouse_drag_callbacks.append
    def get_coord_fg(layer,event):
        if layer.mode == 'add':
            coord = make_int_rev(layer.coordinates)
            if coord not in coord_feat_dict.keys():
                # layer.remove_selected()
                print('Warning: Coord {} not found in feature list, maybe no tissue tile?'.format(coord))
            # else:
            shape = create_square(
                coord[0], coord[1],
                tile_size
            )
#         print(shape)
            shapes_layer_fg.add(shape)
    #         print(shape)
    
    
    @points_layer_fg.bind_key('Backspace')
    @points_layer_fg.bind_key('Delete')
    def delete_selected(layer):
        """Delete all selected points."""
        if str( layer._mode) in ('select', 'add'):
            print(points_layer_fg.selected_data)
            shapes_layer_fg.selected_data = points_layer_fg.selected_data
            shapes_layer_fg.remove_selected()
            layer.remove_selected()
    

    

    
    layer_tiles = viewer.add_shapes(grid_tissue, 
                                      shape_type='polygon',
                                      face_color= 'transparent',
                                      edge_width=10,
                                  edge_color='gray',
    #                                   z_index=0,
                                      opacity = 0.5
                                     )
    

    
    @viewer.bind_key('Control-r', overwrite = True)
    def classify(viewer):
        #todo better indexing of selected features rather than recalculation?
        print('classifying')

        print('fg samples (points)',len(points_layer_fg.data))
        print('bg samples (points)',len(points_layer_bg.data))

        print('fg samples (tiles)', len(shapes_layer_fg.data))
        print('bg samples (tiles)', len(shapes_layer_bg.data))

        print('removing redundant tiles ...')

        set_fg = [i for i, coord in enumerate(viewer.layers['points_fg'].data) if make_int_rev(coord) not in coord_feat_dict.keys()]
        set_bg = [i for i, coord in enumerate(viewer.layers['points_bg'].data) if make_int_rev(coord) not in coord_feat_dict.keys()]
        shapes_layer_fg.selected_data = set(set_fg)
        shapes_layer_fg.remove_selected()
        shapes_layer_bg.selected_data = set(set_bg)
        shapes_layer_bg.remove_selected()
        points_layer_fg.selected_data = set(set_fg)
        points_layer_fg.remove_selected()
        points_layer_bg.selected_data = set(set_bg)
        points_layer_bg.remove_selected()

        print('fg samples (points)', len(points_layer_fg.data))
        print('bg samples (points)', len(points_layer_bg.data))

        print('fg samples (tiles)', len(shapes_layer_fg.data))
        print('bg samples (tiles)', len(shapes_layer_bg.data))

        features_selected_fg = np.array(
            [coord_feat_dict[make_int_rev(coord)] for coord in viewer.layers['points_fg'].data]
        )
        features_selected_bg = np.array(
            [coord_feat_dict[make_int_rev(coord)] for coord in viewer.layers['points_bg'].data]
        )
        
        clf = LogisticRegression(random_state=0).fit(
        np.concatenate(
            (features_selected_bg,features_selected_fg),axis = 0
        ),
        [0]*len(features_selected_bg)+[1]*len(features_selected_fg)
    )
        preds = clf.predict(features)
        tiles_pred = np.array([grid_tissue[i] for i,pred in enumerate(preds) if pred == 1])
        print('positive tiles', len(tiles_pred))
        #todo update layer if exsts
        layer_names =[l.name for l in viewer.layers]
        if 'tiles_pred' not in layer_names:
            layer_tiles_pred = viewer.add_shapes(tiles_pred, 
                                          shape_type='polygon',
                                          face_color= 'red',
                                          edge_width=10,
                                      edge_color='red',
                                        opacity=0.2,
                                              z_index=1)
        else:
            viewer.layers['tiles_pred'].data = tiles_pred
           

