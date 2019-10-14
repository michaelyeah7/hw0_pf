import matplotlib.pyplot as plt
import numpy as np

def load_data(fileName, skip_header, skip_footer, cols):
    print("filename",fileName)
    data = np.genfromtxt(fileName,skip_header = skip_header, \
        skip_footer = skip_footer, names=True, dtype=None, delimiter=' ', usecols=cols)
    return data

def plot_trajectory_Q2(trajectory):
    """plot positions along a given trajectory
    
    Args:
        trajectory: list of positions, position [x, y, theta]
    """
    arrow_length = 0.1
    x = []
    y = []
    for pos in trajectory:
        x.append(pos[0])
        y.append(pos[1])
        # arrow_dx = arrow_length*np.cos(pos[2])
        # arrow_dy = arrow_length*np.sin(pos[2])
        # plt.arrow(pos[0], pos[1], arrow_dx, arrow_dy, \
        #     shape='full', length_includes_head=False, head_width=.01, head_starts_at_zero = True, color = 'r')
    plt.plot(x, y)
    
    plt.savefig('./results/trajectory_Q2.png')

def plot_trajectory_Q3(trajectory):
    """plot positions along a given trajectory
    
    Args:
        trajectory: list of positions, position [x, y, theta]
    """
    x = []
    y = []
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x.append(pos[0])
        y.append(pos[1])  
        i += 100  
    plt.plot(x, y)
    
    plt.savefig('./results/trajectory_Q3.png')

def plot_trajectory_Q3_ground_truth(x, y):
    """plot ground truth trajectory
    
    Args:
        x: x coordinates
        y: y coordinates
    """ 
    plt.plot(x, y)
    
    plt.savefig('./results/trajectory_Q3_ground_truth.png')

def plot_predict_trajectory_Q7(trajectory):
    x = []
    y = []
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x.append(pos[0])
        y.append(pos[1])  
        i += 100  
    plt.plot(x, y)
    
    plt.savefig('./results/predict_trajectory_Q7.png')    

def generate_barcode_dict(barcode_data):
    barcode_dict = {}
    for i in range(5,len(barcode_data)):
        barcode_dict[barcode_data[i][1]] = barcode_data[i][0]
    return barcode_dict

def generate_landmark_gt_dict(landmark_gt_data):
    landmark_gt_dict = {}
    for i in range(len(landmark_gt_data)):
        landmark_gt_dict[landmark_gt_data[i][0]] = [landmark_gt_data[i][1], landmark_gt_data[i][1]]
    return landmark_gt_dict

def whether_landmark(barcode):
    barcodes = [63,25, 45, 16, 61, 36, 18, 9, 72, 70, 81, 54, 27, 7, 90]
    return barcode in barcodes