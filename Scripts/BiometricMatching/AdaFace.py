#Import the libraries
#This code generated attribute edited images of a particular attribute given in the test_image_path
#Please change the test_image_path for every attribute
import net
import torch
import os
from face_alignment import align
import numpy as np
import csv


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    if(np_img.shape!=(112,112,3)):
        pass
    else:
        #print("np.img",np_img.shape)
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
        return tensor
 

if __name__ == '__main__':
    list_dictionary = {
        #lists to store the genuine and imposter scores
        'Lg_orig':[],  #genuine list
        'Li_orig':[]   #impostor list
        }
    model = load_pretrained_model('ir_50')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = '/path/to/folder/which/has/images/edited/with/particular/attribute/eg:bangs'
    orig_path = '/path/to/original/images'
    folders = os.listdir(orig_path)
    for folder in folders:
            folder_path = os.path.join(orig_path,folder)
            if os.path.isdir(folder_path):
                features_orig=[]
                orig_files_list = os.listdir(folder_path) 
                for o in orig_files_list:
                    if os.path.isfile(folder_path+'/'+o):
                        path_1 = os.path.join(folder_path, o)
                        aligned_rgb_img_1 = align.get_aligned_face(path_1)
                        bgr_tensor_input_1 = to_input(aligned_rgb_img_1)
                        if(bgr_tensor_input_1)==None:
                            continue
                        featureo, _ = model(bgr_tensor_input_1)
                        features_orig.append(featureo)
                M = torch.cat(features_orig).detach().numpy()
                for fname in sorted(os.listdir(test_image_path)):
                    path = os.path.join(test_image_path, fname)
                    aligned_rgb_img = align.get_aligned_face(path)
                    bgr_tensor_input = to_input(aligned_rgb_img)
                    if(bgr_tensor_input)==None:
                        continue
                    feature, _ = model(bgr_tensor_input)
                    N = feature.detach().numpy()
                    E = list(np.sum((np.multiply(M,N)),axis = 1))
                    E = list(np.round(E,decimals=4))
                    if ((int(str(fname[3:7])))==int(str(folder[4:7]))):
                        list_dictionary['Lg_orig'].extend(E)
                    else:
                        list_dictionary['Li_orig'].extend(E)
          
    with open('/path/to/new/csv/file/to/store/scores','w',newline='') as filec:
        writer = csv.writer(filec)
        for m in list_dictionary.values():
            writer.writerow(m)

                
                        
                        
                
                
                
        
        
    
