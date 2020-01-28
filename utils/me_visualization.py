import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pdb
import scipy.io

class me_visualization:
    def __init__(self):
        self.initialization()
        self.get_drivers_data()
        for driver_name in self.driver_names:
            dir_gen_video= os.path.join(self.dir_fig_folder, driver_name+ '.avi')
            print('[i]  Driver %s in processing.'%(driver_name))
            self.prepare_data(driver_name)
            self.visualization(driver_name)

    def initialization(self):
        self.dir_mat_data= os.path.join('/mnt', 'sdc1', 'ysshin', 'Hyundai', 'data')
        self.dir_video_data= os.path.join('/mnt', 'sdc1', 'ysshin', 'Hyundai', 'videos')

        self.dir_fig_folder= os.getcwd()+ '/figs'
        if not os.path.exists(self.dir_fig_folder):
            os.mkdir(self.dir_fig_folder)
        
        self.value_names= ['WHL_SPD_RR_t35', 'SAS_Angle_t34']
        self.bbox_names= ['Num_Obstacles_t36',
                          'Obstacle_Valid_t37', 'Obstacle_Type_t37',
                          'Obstacle_Status_t37', 'Obstacle_Rel_Vel_X_t37',
                          'Obstacle_Pos_Y_t37', 'Obstacle_Pos_X_t37', 
                          'Obstacle_ID_t37',
                          'Obstacle_Valid_t38', 'Obstacle_Type_t38',
                          'Obstacle_Status_t38', 'Obstacle_Rel_Vel_X_t38',
                          'Obstacle_Pos_Y_t38', 'Obstacle_Pos_X_t38',
                          'Obstacle_ID_t38',
                          'Obstacle_Valid_t39', 'Obstacle_Type_t39',
                          'Obstacle_Status_t39', 'Obstacle__Rel_Vel_X_t39',
                          'Obstacle_Pos_Y_t39', 'Obstacle_Pos_X_t39',
                          'Obstacle_ID_t39',
                          'Obstacle_Valid_t40', 'Obstacle_Type_t40',
                          'Obstacle_Status_t40', 'Obstacle_Rel_Vel_X_t40',
                          'Obstacle_Pos_Y_t40', 'Obstacle_Pos_X_t40',
                          'Obstacle_ID_t40',
                          'Radar_Vel_X_t42', 'Radar_Pos_X_t42',
                          'Radar_Match_Confidence_t42',
                          'Obstacle_Width_t42', 'Obstacle_Length_t42',
                          'Obstacle_Lane_t42', 'Obstacle_Age_t42',
                          'Matched_Radar_ID_t42']
        self.num_test_frames= 2000

    def get_drivers_data(self):
        self.driver_names= os.listdir(self.dir_video_data)
        # extension delete
        for driver_idx, driver_name in enumerate(self.driver_names):
            self.driver_names[driver_idx]= driver_name.split('.')[0]
        # Check each name of video data exists in mat folder
        for data_name in self.driver_names:
            data_name= data_name.split('.')[0]
            data_name= data_name+ '.mat'
            if not (data_name in os.listdir(self.dir_mat_data)):
                raise KeyError('[ERROR]  There is no mat file named %s'%(data_name))

    def prepare_data(self, driver_name):
        self.signal_values= {}
        self.signal_bboxes= {}
        self.signal_length= {}
        self.ts= {}
        self.text_offset= 12*(len(self.value_names))+ 2
        dir_mat_data= os.path.join(self.dir_mat_data, driver_name+ '.mat')
        dir_video_data= os.path.join(self.dir_video_data, driver_name+ '.avi')
        self.data_mat= scipy.io.loadmat(dir_mat_data)
        self.data_video= cv2.VideoCapture(dir_video_data)
        # signal, time dictionary
        for s_name in self.data_mat.keys():
            t_name= s_name.split('_')[-1]
            if s_name in self.value_names:
                self.signal_values[s_name]= self.data_mat[s_name]
                self.signal_length[s_name]= len(self.data_mat[s_name])
                if t_name not in self.ts.keys():
                    self.ts[t_name]= self.data_mat[t_name]
            elif s_name in self.bbox_names:
                self.signal_bboxes[s_name]= self.data_mat[s_name]
                self.signal_length[s_name]= len(self.data_mat[s_name])
                if t_name not in self.ts.keys():
                    self.ts[t_name]= self.data_mat[t_name]
        self.num_frames= self.data_video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps= self.data_video.get(cv2.CAP_PROP_FPS)

    def show_values(self, frame, frame_count):
        font_face= cv2.FONT_HERSHEY_SIMPLEX
        scale= 0.3
        color= (255, 255, 255)
        for s_idx, s_name in enumerate(self.value_names):
            sync_factor= self.signal_length[s_name]/ self.num_frames
            if s_name== 'WHL_SPD_RR_t35':
                text= 'WHL SPD : %.4f'%(self.signal_values[s_name][int(frame_count*sync_factor)][0])
                text_size= cv2.getTextSize(text, font_face, scale, 1)
                pos= (3, self.org_height+ text_size[0][1]+ 2)
                cv2.putText(frame, str(text), pos, font_face, scale, color, 1, cv2.LINE_AA)
            elif s_name== 'SAS_Angle_t34':
                text= 'SAS Angle : %.4f'%(self.signal_values[s_name][int(frame_count*sync_factor)][0])
                text_size= cv2.getTextSize(text, font_face, scale, 1)
                pos= (self.org_width- text_size[0][0] , self.org_height+ text_size[0][1]+ 2)
                cv2.putText(frame, str(text), pos, font_face, scale, color, 1, cv2.LINE_AA)
        return frame

    def show_map(self, frame, frame_count):
        template= np.zeros(shape= (self.extra_width, self.ext_height, 3))
        # 도로 폭 대략 3m
        load_offset= 3
        template[100, :]= (255, 255, 255)
        template= np.transpose(template, (1, 0, 2))
        frame= np.append(frame, template, axis=1)
        frame= np.append(frame, f_arr[:, :, 0:3], axis=1)
        return frame

    def visualization(self, driver_name):
        self.extra_height= 30
        self.extra_width= 200
        self.org_height= 180
        self.org_width= 320
        self.ext_height= self.org_height+ self.extra_height
        self.ext_width= self.org_width+ self.extra_width
        self.img_channels= 3
        frame_count= 0
        fourcc= cv2.VideoWriter_fourcc(*'DIVX')
        gen_video= cv2.VideoWriter(self.dir_fig_folder+ '/%s.avi'%(driver_name), fourcc, 30, (self.ext_width, self.ext_height))
        plt.switch_backend('agg')
        while frame_count< self.num_test_frames:
            ret, frame= self.data_video.read()
            extra_template= np.zeros(shape= (self.extra_height, self.org_width, self.img_channels))
            frame_expanded= np.append(frame, extra_template, axis= 0)
            frame_expanded= self.show_values(frame_expanded, frame_count)
            frame_expanded= self.show_map(frame_expanded, frame_count).astype(np.uint8)

            gen_video.write(frame_expanded)
            if frame_count%300 == 0:
                progress= frame_count/ self.num_test_frames* 100
                print('[i]    progress %.4f %%'%(progress))
            frame_count+= 1
        cv2.destroyAllWindows()
        gen_video.release()

me_visualization()

