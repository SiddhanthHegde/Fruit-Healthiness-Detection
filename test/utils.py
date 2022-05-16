import os
import json
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from skimage import io
from multiprocessing import Pool, cpu_count
import torch
from osgeo import ogr

def get_image_dimensions(rgb_path: str):    
    rgb_raster = io.imread(rgb_path)
    height, width = rgb_raster.shape[0:2]  
    return height, width

def create_task_list(image_path, out_dir, save_id, height, width, stride, dim):
    task_list = []
    for i in range(0, height + 1, stride):
        for j in range(0, width + 1, stride):
            task_list.append([image_path, out_dir, save_id, i, j, dim])
            save_id += 1

    return task_list, save_id

def chip_test_images(images_dir: str, out_dir: str, stride: str, dim: int):
    save_id = 0
    image_names = os.listdir(images_dir)
    image_to_id = {}

    for image_name in tqdm(image_names):
        image_path = os.path.join(images_dir, image_name)
        height, width = get_image_dimensions(image_path)
        image_to_id[image_name] = save_id
        task_list, save_id = create_task_list(image_path, out_dir, save_id, height, width, stride, dim)

        p = Pool(cpu_count() - 1)
        p.starmap(translate_single, task_list)
        p.close()
        p.join()
    
    json_file = os.path.join(os.path.dirname(out_dir), 'batching_numbers.json')        
    json.dump(image_to_id, open(json_file, 'w'))

    return json_file

def translate_single(image_path, out_dir, save_id, i, j, dim):
    out_chip_path = os.path.join(out_dir, '{}.png'.format(save_id))
    gdal.Translate(out_chip_path, image_path, srcWin=[i, j, dim, dim], format="PNG", outputType=gdal.GDT_Byte)

    if os.path.exists(out_chip_path + '.aux.xml'):
        os.remove(out_chip_path + '.aux.xml')

def predict_and_save(model, dataloader, save_dir, device):

    with torch.no_grad():
        for _, batch in tqdm(list(enumerate(dataloader))):
            
            test_images = batch['image'].to(device)
            names = batch['image_name']
            scores = model.forward(test_images)
            _, preds = torch.max(scores, dim=1)
            preds = preds.type(torch.uint8)

            for name, pred in zip(names, preds):
                np_img = pred.detach().cpu().numpy()
                save_path = os.path.join(save_dir, name)
                io.imsave(save_path, np_img)

def stitch_and_save(json_path, height, width, stride, window_size, output_dir, final_pred_dir):
    chip_numbers_dict = json.load(open(json_path))
    for image_name, chip_id in chip_numbers_dict.items():
        empty_pred = np.zeros((height + window_size, width + window_size), dtype=np.uint8)
        for i in range(0, height + 1, stride):
            for j in range(0, width + 1, stride):
                chip_path = os.path.join(output_dir, str(chip_id) + '.png')
                chip_pred = io.imread(chip_path)
                chip_id += 1
                empty_pred[j:j+window_size, i:i+window_size] = chip_pred

        final_pred = empty_pred[:-window_size, :-window_size]
        final_pred_path = os.path.join(final_pred_dir, image_name.replace('.JPG','.png'))
        io.imsave(final_pred_path, final_pred)

def polygonize_raster(img_path, out_path):
    src_ds = gdal.Open(img_path)
    srcband = src_ds.GetRasterBand(1)

    dst_layername = out_path
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )

    gdal.Polygonize( srcband, srcband, dst_layer, -1, [], callback=None )  ##
    dst_ds.Destroy()
    src_ds=None

def rasterize(shp_vector_path, reference_tif_path, out_tif_path):
    ds = gdal.Open(reference_tif_path)
    img = ds.GetRasterBand(1).ReadAsArray()
    h, w = img.shape
    ds = None
    input_shp = ogr.Open(shp_vector_path)
    shp_layer = input_shp.GetLayer()
    shp_vector_name = shp_layer.GetName()
    raster = gdal.Open(reference_tif_path)
    gt =raster.GetGeoTransform()
    pixelSizeX = gt[1]
    pixelSizeY =-gt[5]
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * raster.RasterXSize
    miny = maxy + gt[5] * raster.RasterYSize
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    cmd = "gdal_rasterize -burn 1 -ot Float32 -te {} {} {} {} -ts {} {} -l {} {} {}".format(minx, miny, maxx, maxy, w, h, shp_vector_name, shp_vector_path, out_tif_path)
    os.system(cmd)
    raster = None
    return out_tif_path