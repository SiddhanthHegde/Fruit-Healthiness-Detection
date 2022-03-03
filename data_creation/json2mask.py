import os
import json
import argparse
import numpy as np
from labelme.utils import shape_to_mask
from PIL import Image
from multiprocessing import Pool
from defaults_data_creation import _C as cfg

#-----------------------------------------------------------------------

def json_to_mask(root_dir: str, filename: str, save_dir: str):

    '''
    Parameters:

    root_dir : Path to the root directory containing all the jsons
    filename : Filename of each json
    save_dir : Path to the output directory to save the masks in png format

    '''
    
    file_path = os.path.join(root_dir, filename)
    json_dict = json.load(open(file_path))

    height = json_dict['imageHeight']
    width = json_dict['imageWidth']
    polygons = json_dict['shapes']
    n_polys = len(json_dict['shapes'])

    healthy_mask = np.zeros((height,width),dtype=np.uint8)
    infected_mask = np.zeros((height,width),dtype=np.uint8)
    ternary_mask = np.zeros((height,width),dtype=np.uint8)

    for i in range(n_polys):

        label = polygons[i]['label'] 

        # Label 1 corresponds to the outline of the fruit
        if label == '1':
            temp_mask = shape_to_mask(
                (height,width),
                polygons[i]['points'],
                shape_type=None,
                line_width=1,
                point_size=1
            )
            healthy_mask[np.where(temp_mask)] = 1

        # Label 2 corresponds to the outlines of the infected regions
        elif label == '2':
            temp_mask = shape_to_mask(
                (height,width),
                polygons[i]['points'],
                shape_type=None,
                line_width=1,
                point_size=1
            )
            infected_mask[np.where(temp_mask)] = 1
        
        # Label 3 corresponds to the outlines of healthy regions inside the infected regions
        elif label == '3':
            temp_mask = shape_to_mask(
                (height,width), 
                polygons[i]['points'],
                shape_type=None,
                line_width=1,
                point_size=1
            )
            ternary_mask[np.where(temp_mask)] = 1

        else:
            print(f'Label {label} cannot be used, required range - (0-2)')
            return

    healthy_mask[np.where(infected_mask == 1)] = 2
    healthy_mask[np.where(ternary_mask == 1)] = 1

    save_path = os.path.join(
        save_dir,
        filename.replace('.json','.png')
    )

    Image.fromarray(healthy_mask).save(save_path)

#-----------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Script to convert jsons into mask of png')
    parser.add_argument('--cfg', metavar = 'FILE', help = 'path to config file', type = str)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

#-----------------------------------------------------------------------

    assert cfg.JSON2MASK.num_labels == 3, 'There should exist only 3 labels'

    root_dir = cfg.JSON2MASK.root_dir
    save_dir = cfg.JSON2MASK.save_dir
    num_processes = cfg.JSON2MASK.num_processes
    
    os.makedirs(save_dir,exist_ok=True)
    task_list = []
    json_list = [x for x in os.listdir(root_dir) if x.endswith('.json')]

    print('Converting JSONs to masks !!!')
    print(f'Found a total of {len(json_list)} jsons in the root directory')

    for filename in json_list:
        task_list.append([root_dir,filename,save_dir])

    p = Pool(num_processes)
    p.starmap(json_to_mask, task_list)
    p.close()
    p.join()

    print('Successfully converted all jsons to masks !!!')