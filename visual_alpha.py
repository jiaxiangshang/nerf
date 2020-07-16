#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jiaxiang Shang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: jiaxiang.shang@gmail.com
@time: 6/24/20 10:24 AM
@desc:
'''

# system
from __future__ import print_function

import os
import sys
import argparse, ast

#
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

# self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tool_data_dir = os.path.dirname(_cur_dir) # ../
_deep_learning_dir = os.path.dirname(_tool_data_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

parser = argparse.ArgumentParser(description='Preprocess Altizure Pipline')

# 0.
parser.add_argument('--dic_test', type=str, default='/home/jshang/SHANG_Data/8_cvpr2021_trainingData/0_nerf_origin/logs_0_nerf_llff_data/rs_paper_fern_div/testset_001000', help='')


PARSER = parser.parse_args()

"""
"""

if __name__ == '__main__':
    #
    if os.path.isdir(PARSER.dic_test):
        pass
    else:
        print("No such dictionary", PARSER.dic_test)

    # Collect all file
    list_name_files = []
    for (dirpath, dirnames, filenames) in os.walk(PARSER.dic_test):
        list_name_files = filenames
        break  # pass 0
    list_name_files_gz = [lnf for lnf in list_name_files if lnf.find('.npz') != -1]

    for i in range(len(list_name_files_gz)):
        name_alpha = list_name_files_gz[i]

        name_pure = name_alpha[:3]

        name_render = name_pure + '.png'

        #
        path_alpha = os.path.join(PARSER.dic_test, name_alpha)
        path_render = os.path.join(PARSER.dic_test, name_render)

        #
        alpha = np.load(path_alpha)['alpha']
        alpha_0 = np.load(path_alpha)['alpha_0']
        image = cv2.imread(path_render)

        #
        def draw_rectangle(event, x, y, flags, param):
            global ix, iy
            if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = x, y
                print(alpha.shape, x, y)
                alpha_coarse = alpha_0[iy, ix]
                alpha_192 = alpha[iy, ix]

                plt.subplot(2, 1, 1)
                plt.plot(range(len(alpha_coarse)), alpha_coarse, color="blue", linewidth=1, markerfacecolor='black', marker='o',)
                plt.subplot(2, 1, 2)
                plt.plot(range(len(alpha_192)), alpha_192, color="black", linewidth=1, markerfacecolor='black', marker='o', )

                plt.xlabel('Depth')
                plt.ylabel('Alpha')

                plt.show()

                # def get_img_from_fig(fig, dpi=180):
                #     buf = io.BytesIO()
                #     fig.savefig(buf, format="png", dpi=dpi)
                #     buf.seek(0)
                #     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                #     buf.close()
                #     img = cv2.imdecode(img_arr, 1)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #
                #     return img
                #
                # # you can get a high-resolution image as numpy array!!
                # plot_img_np = get_img_from_fig(fig)

                # while (1):
                #     cv2.imshow('alpha', plot_img_np)
                #     if cv2.waitKey(20) & 0xFF == 27:
                #         break


        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_rectangle)
        while (1):
            cv2.imshow('image', image)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()