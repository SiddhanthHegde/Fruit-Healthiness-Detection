import os
import shutil
from skimage import io
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import JOIN_STYLE, Polygon, MultiPolygon
import geopandas as gpd
from utils import polygonize_raster, rasterize

def close_holes(poly: Polygon) -> Polygon:
        if poly.interiors:
            return Polygon(list(poly.exterior.coords))
        else:
            return poly

def remove_tiny_areas(final_pred_dir):

    image_names = [x for x in os.listdir(final_pred_dir) if not x.endswith('.tif') and not x.endswith('coloured.png')]
    shp_dir = 'test/data/shps'
    if os.path.exists(shp_dir):
        shutil.rmtree(shp_dir)
    os.makedirs(shp_dir)
    inf_dict = {}

    for image_name in image_names:
        image_path = os.path.join(final_pred_dir, image_name)
        image = io.imread(image_path)

        #---------creating shp file for only health---------------------------------
        only_health = image.copy()
        only_health[np.where(only_health == 2)] = 1
        only_health_save = os.path.join(shp_dir, 'healthy.png')
        io.imsave(only_health_save, only_health.astype(np.uint8))
        polygonize_raster(only_health_save, only_health_save.replace('.png','.shp'))

        #---------finding the polygon with max area---------------------------------
        h_df = gpd.read_file(only_health_save.replace('.png','.shp'))
        max_area = -1
        save_id = 0
        for id, row in h_df.iterrows():
            if not row['geometry'].is_valid:
                row['geometry'] = row['geometry'].buffer(0)

            if row['geometry'].area > max_area:
                max_area = row['geometry'].area
                save_id = id
        other_ids = [x for x in range(len(h_df)) if x != save_id]
        h_df.drop(other_ids, axis = 0, inplace=True)

        #---------Filling the polygon with only health to remove bg inside fruit---
        h_df = h_df.geometry.apply(lambda p: close_holes(p))
        h_df.to_file(only_health_save.replace('.png','.shp'))

        #---------only healthy is ready and is converted into raster---------------
        h_df = gpd.read_file(only_health_save.replace('.png','.shp'))
        rasterize(only_health_save.replace('.png','.shp'), image_path, image_path.replace('.png', '_health.tif'))

        #---------creating shp file for only infection-----------------------------
        only_inf = image.copy()
        only_inf[np.where(only_inf == 1)] = 0
        only_inf[np.where(only_inf == 2)] = 1
        only_inf_save = os.path.join(shp_dir, 'inf.png')
        io.imsave(only_inf_save, only_inf.astype(np.uint8))
        polygonize_raster(only_inf_save, only_inf_save.replace('.png','.shp'))

        #---------calculate areas of infection and remove infection outside fruit---
        i_df = gpd.read_file(only_inf_save.replace('.png','.shp'))
        inf_area = 0
        ids = []
        healthy_poly = h_df.iloc[0]['geometry']

        for id, row in i_df.iterrows():
            if not row['geometry'].is_valid:
                row['geometry'] = row['geometry'].buffer(0)
            
            if not healthy_poly.contains(row['geometry']):
                ids.append(id)
            inf_area += row['geometry'].area
        
        i_df.drop(ids, axis=0, inplace=True)

        #---------fill in the infection areas which has holes------------------------
        i_df = i_df.geometry.apply(lambda p: close_holes(p))
        i_df.to_file(only_inf_save.replace('.png','.shp'))

        rasterize(only_inf_save.replace('.png','.shp'), image_path, image_path.replace('.png', '_inf.tif'))
        new_img = io.imread(image_path.replace('.png', '_health.tif'))
        only_inf = io.imread(image_path.replace('.png', '_inf.tif'))
        new_img[np.where(only_inf == 1)] = 2
        
        #---------Colouring image and saveing in rgb form----------------------------
        save_img = np.stack([new_img,new_img,new_img], axis=2)
        save_img[np.all(save_img == (1, 1, 1), axis=-1)] = (0,255,0)
        save_img[np.all(save_img == (2, 2, 2), axis=-1)] = (255,0,0)
        io.imsave(image_path.replace('.png', '_coloured.png'), save_img.astype(np.uint8))

        for x in os.listdir(shp_dir):
            os.remove(os.path.join(shp_dir,x))

        inf_dict[image_name] = inf_area / max_area


if __name__ == '__main__':
    final_pred_dir = 'test/data/final_preds'
    remove_tiny_areas(final_pred_dir)