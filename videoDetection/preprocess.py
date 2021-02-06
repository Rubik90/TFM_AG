import os
import argparse
import csv 
import sys
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--src_vid_path', default='./videos/', type=str) # Define the path where the videos are stored
parser.add_argument('--fe_bin_path', default='./FeatureExtraction', type=str) # Define OpenFace FeatureExtraction binary
parser.add_argument('--flv_bin_path', default='./FaceLandmarkVid', type=str) # Define OpenFace FeatureExtraction binary
parser.add_argument('--orig_out_data_path', default='./processed/', type=str) # Define the path where to store the output data
parser.add_argument('--dest_out_data_path', default='./features/', type=str) # Define the path where to store the resulting features
args = parser.parse_args()

vid_list = os.listdir(args.src_vid_path)
vid_list.sort()


for i in range(0, len(vid_list)):
    print(vid_list[i][:-4])
    
    # Feature Extraction
    print(args.fe_bin_path + ' -f ' + args.src_vid_path+vid_list[i])
    os.system(args.fe_bin_path + ' -f '+ args.src_vid_path+vid_list[i])
    
    # Face Landmark
    print(args.flv_bin_path + ' -f ' + args.src_vid_path+vid_list[i])
    os.system(args.flv_bin_path + ' -f '+ args.src_vid_path+vid_list[i])
    
    # Remove previously extraced features
    if os.path.exists(args.dest_out_data_path+'aligned_face/'+vid_list[i][:-4]+'_aligned'):
        shutil.rmtree(args.dest_out_data_path+'aligned_face/'+vid_list[i][:-4]+'_aligned')

    src_path = args.orig_out_data_path + vid_list[i][:-4]+'_aligned'
    
    face_img_path = args.dest_out_data_path+'aligned_face/'+vid_list[i][:-4]+'_aligned/'
    shutil.move(src_path, face_img_path)
		
    # Create CSV file for indexing aligned images
    align_imglist = os.listdir(face_img_path)	
    align_imglist.sort()
    csv_path = args.dest_out_data_path+ 'csv_files/' 
	
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
		
    out_csv_path = csv_path + vid_list[i][:-4] + '.csv'
    with open(out_csv_path, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',')

        for j in range(0, len(align_imglist)):
            data_writer.writerow([args.dest_out_data_path + 'aligned_face/' + vid_list[i][:-4]+'_aligned/' + align_imglist[j]])
