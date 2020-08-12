#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:


import numpy as np
import napari
import cv2
import matplotlib.pyplot as plt
from IPython.core.display import display
from tqdm import tqdm
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# In[3]:


import napari_lazy_openslide


# In[4]:


import openslide


# In[5]:


path = '/home/dhsr/RD/S12-13045-1-A-9_12-2414.scn'
# path = '/home/reasat/data/dhsr/RD/S12-10712-1-B-2_12-2010.scn'
# path = '/home/reasat/data/dhsr/RD/S12-13562-1-A-6_12-2529.scn'


# In[6]:


wsi = openslide.OpenSlide(path)


# In[7]:


dict(wsi.properties)


# In[8]:


regions_dict = {key: wsi.properties[key] for key in wsi.properties.keys() if 'openslide.region' in key}
regions_dict


# In[9]:


# reform
regions_dict_rf = {}
for i in range(len(regions_dict)//4):
    regions_dict_rf['region-{}'.format(i)]={
        'x':int(regions_dict['openslide.region[{}].x'.format(i)]),
        'y':int(regions_dict['openslide.region[{}].y'.format(i)]),
        'width':int(regions_dict['openslide.region[{}].width'.format(i)]),
         'height': int(regions_dict['openslide.region[{}].height'.format(i)])
    }
        


# In[10]:


regions_dict_rf


# In[11]:


img, prop = napari_lazy_openslide.lazy_openslide.reader_function(path)[0]


# In[12]:


prop


# In[13]:


img


# In[14]:


for d in img:
    display(d)


# # Viewer

# In[15]:


def create_square(x,y,tile_size):
    return [[y,x],
            [y+tile_size,x],
            [y+tile_size,x+tile_size],
            [y,x+tile_size]
           ]


# In[16]:


X_OFF = regions_dict_rf['region-0']['x']
Y_OFF = regions_dict_rf['region-0']['y']


# In[19]:


viewer = napari.view_image(img)


# In[18]:


# viewer.add_image(img)


# In[35]:


layer_img = viewer.layers['img']


# In[20]:


tile_size = 1024


# In[43]:


points_layer_bg = viewer.add_points(
    name='points_bg',
    size = 50,
    face_color= 'blue'
)
shapes_layer_bg = viewer.add_shapes(name='tiles_bg',
                                    edge_color='blue',
                                   face_color='blue',
                                   opacity=0.3)

@points_layer_bg.mouse_drag_callbacks.append
def get_coord_bg(layer,event):
    if layer.mode == 'add':
        coord = layer.coordinates
        shape = create_square(
        X_OFF+(coord[1]-X_OFF)//tile_size*tile_size,
        Y_OFF+(coord[0]-Y_OFF)//tile_size*tile_size, 
        tile_size
    )
       
        shapes_layer_bg.add(shape)
#         layer.add(coord)
    


# In[44]:


@points_layer_bg.bind_key('Backspace')
@points_layer_bg.bind_key('Delete')
def delete_selected(layer):
    """Delete all selected points."""
    if str( layer._mode) in ('select', 'add'):
        shapes_layer_bg.selected_data = points_layer_bg.selected_data
        shapes_layer_bg.remove_selected()
        layer.remove_selected()


# In[45]:


points_layer_fg = viewer.add_points(
    name='points_fg', 
    size=50, 
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
        coord = layer.coordinates
#         print(coord)
        shape = create_square(
        X_OFF+(coord[1]-X_OFF)//tile_size*tile_size,
        Y_OFF+(coord[0]-Y_OFF)//tile_size*tile_size, 
        tile_size
    )
#         print(shape)
        shapes_layer_fg.add(shape)
#         print(shape)


# In[46]:


@points_layer_fg.bind_key('Backspace')
@points_layer_fg.bind_key('Delete')
def delete_selected(layer):
    """Delete all selected points."""
    if str( layer._mode) in ('select', 'add'):
        shapes_layer_fg.selected_data = points_layer_fg.selected_data
        shapes_layer_fg.remove_selected()
        layer.remove_selected()


# # Create Grid

# In[25]:


# tiles = []
x_list = []
y_list = []
for key in regions_dict_rf.keys():
    region = regions_dict_rf[key]
    y = np.arange(region['y'],
                  region['y']+region['height']//tile_size*tile_size,
                  tile_size)
    x = np.arange(region['x'],
                  region['x']+region['width']//tile_size*tile_size,
                  tile_size)
    x_list.append(x)
    y_list.append(y)

yv, xv = np.meshgrid(np.concatenate(np.array(y_list)),
                     np.concatenate(np.array(x_list)))
xv = xv.flatten()
yv = yv.flatten()
# for x,y in zip(xv,yv):
#     tiles.append(data[x:x+tile_size,y:y+tile_size])
# len(tiles) 


# In[26]:


len(xv)


# In[27]:


def create_square(x,y,tile_size):
    return [[y,x],
            [y+tile_size,x],
            [y+tile_size,x+tile_size],
            [y,x+tile_size]
           ]


# In[28]:


def has_tissue(wsi, x,y,tile_size,downsample_level=1):
    tile = wsi.read_region(
            location= (np.uint32(x),np.uint32(y)),
            level = downsample_level,
            size = (np.uint32(tile_size//wsi.level_downsamples[downsample_level]),
                   np.uint32(tile_size//wsi.level_downsamples[downsample_level]))
        )
    tile = np.array(tile)[:,:,:3]
    mean = tile.reshape((-1,3)).mean(axis=0)
    return ((mean<np.array([225]*3)).all())


# In[29]:


#triangle = np.array([[11, 13], [111, 113], [22, 246]])
grid_tissue = [create_square(x,y,tile_size) for x,y in zip(tqdm(xv),yv)           if has_tissue(wsi, x,y,tile_size)]
print(len(grid_tissue))


# In[30]:


layer_tiles = viewer.add_shapes(grid_tissue, 
                                  shape_type='polygon',
                                  face_color= 'transparent',
                                  edge_width=10,
                              edge_color='gray',
#                                   z_index=0,
                                  opacity = 0.5
                                 )


# # Process features

# In[31]:


def feature_extractor(architecture = 'resnet50'):
    if architecture == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model_features = torch.nn.Sequential(*list(model.children())[:-1])
    if architecture == 'inception_v3':
        model = torchvision.models.inception_v3(pretrained=True)
        model_features = torch.nn.Sequential(*list(model.children())[:-2])
    return model_features


# In[32]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[37]:


def extract_features(model,tiles,layer_img, downsample_level=1):
    model.eval()
    features = []
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for square in tqdm(tiles):
        y = square[0][0]
        x = square[0][1]

        tile = wsi.read_region(
            location= (np.uint32(x),np.uint32(y)),
            level = downsample_level,
            size = (np.uint32(tile_size//wsi.level_downsamples[downsample_level]),
                   np.uint32(tile_size//wsi.level_downsamples[downsample_level]))
        )
        tile = np.array(tile)[:,:,:3]
#         print(tile.shape)
#         tile = layer_img.data[0][y:y+tile_size,x:x+tile_size,:3].compute()
        tile = tile/255   
        tile = (tile-means)/stds
        tile = torch.tensor(tile.transpose(2,0,1),dtype = torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model_features(tile)
            features.append(feat.detach().cpu().numpy().squeeze())
    return np.array(features)


# In[38]:


model_features = feature_extractor('resnet50').to(device)
features = extract_features(model_features,grid_tissue,layer_img, downsample_level=0)


# In[39]:


features.shape


# In[40]:


def create_fit_PCA(data, n_components=None):
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)   
    return p


# In[41]:


pca = create_fit_PCA(features, n_components=min(features.shape))
features_pca = pca.transform(features)


# In[42]:


@viewer.bind_key('Control-r', overwrite = True)
def classify(viewer):
    #todo better indexing of selected features rather than recalculation?
    print('classifying')
    print('fg samples',len(points_layer_fg.data))
    print('bg samples',len(points_layer_bg.data))
    grid_selected_fg = [
        create_square(
            X_OFF+(coord[1]-X_OFF)//tile_size*tile_size,
            Y_OFF+(coord[0]-Y_OFF)//tile_size*tile_size, 
            tile_size
        )
        for coord in viewer.layers['points_fg'].data
    ]
    grid_selected_bg = [
        create_square(
            X_OFF+(coord[1]-X_OFF)//tile_size*tile_size,
            Y_OFF+(coord[0]-Y_OFF)//tile_size*tile_size, 
            tile_size
        )
        for coord in viewer.layers['points_bg'].data
    ]
    
    features_selected_bg = pca.transform(extract_features(
        model_features,grid_selected_bg, layer_img))
    features_selected_fg = pca.transform(extract_features(
        model_features,grid_selected_fg, layer_img))
    clf = LogisticRegression(random_state=0).fit(
    np.concatenate(
        (features_selected_bg,features_selected_fg),axis = 0
    ),
    [0]*len(features_selected_bg)+[1]*len(features_selected_fg)
)
    preds = clf.predict(features_pca)
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
        




