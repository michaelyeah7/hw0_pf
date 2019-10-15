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
        print(pos[0],pos[1],pos[2])
    plt.plot(x, y, label='Trajectory following Q2 command')
    plt.title('Q2') 
    plt.legend()   
    plt.savefig('./results/trajectory_Q2.png')

def plot_trajectory_Q3(env_trajectory, gt_trajectory):
    """plot positions along a given trajectory
    
    Args:
        trajectory: list of positions, position [x, y, theta]
    """
    x = []
    y = []
    for i in range(len(env_trajectory)):
        pos = env_trajectory[i]
        x.append(pos[0])
        y.append(pos[1])  
        i += 100  

    x_gt = []
    y_gt = []
    for i in range(50000):
        # pos = gt_trajectory[i]
        x_gt.append(gt_trajectory[i][1])
        y_gt.append(gt_trajectory[i][2])  
        i += 100  

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label='Dead Reckoned')
    ax.plot(x_gt, y_gt, label='Ground_truth')
    plt.title('Q3')
    ax.legend()
    plt.savefig('./results/trajectory_Q3.png')

def plot_predict_trajectory_Q7(robot_trajectory, robot_pos_predict_trajectory, ground_truth):
    x = []
    y = []
    trajectory = robot_trajectory
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x.append(pos[0])
        y.append(pos[1])  
        i += 100  
    # plt.plot(x, y, lable='ground_truth')

    x_pred = []
    y_pred = []
    trajectory = robot_pos_predict_trajectory
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x_pred.append(pos[0])
        y_pred.append(pos[1])  
        i += 100  
    # plt.plot(x, y, 'filter prediction')

    # ground_truth = load_data('./ds1_Groundtruth.dat', 3, 0, [0,4,5])
    x_gt = []
    y_gt = []
    for i in range(23000):
        x_gt.append(ground_truth[i][1])
        y_gt.append(ground_truth[i][2])

    fig = plt.figure()
    ax = plt.subplot(111)
    # ax.plot(x, y, label='odometry')
    ax.plot(x_pred, y_pred, label='filter prediction')
    ax.plot(x_gt, y_gt, label='ground_truth')
    plt.title('Particle Filter')
    ax.legend()
    
    plt.savefig('./results/predict_trajectory_Q7.png')     

def plot_predict_trajectory_Q8(robot_trajectory, robot_pos_predict_trajectory):
    x = []
    y = []
    trajectory = robot_trajectory
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x.append(pos[0])
        y.append(pos[1])  
        i += 1 
    # plt.plot(x, y, lable='ground_truth')

    x_pred = []
    y_pred = []
    trajectory = robot_pos_predict_trajectory
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x_pred.append(pos[0])
        y_pred.append(pos[1])  
        i += 1  
    # plt.plot(x, y, 'filter prediction')

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label='odometry')
    ax.plot(x_pred, y_pred, label='filter prediction')
    # plt.title('Q8 using Q2 odometry')
    plt.title('Q8.1')
    ax.legend()
    
    plt.savefig('./results/predict_trajectory_Q8.png')     

def plot_predict_trajectory_Q9(robot_trajectory, robot_pos_predict_trajectory, ground_truth):
    x = []
    y = []
    trajectory = robot_trajectory
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x.append(pos[0])
        y.append(pos[1])  
        i += 100  
    # plt.plot(x, y, lable='ground_truth')

    x_pred = []
    y_pred = []
    trajectory = robot_pos_predict_trajectory
    for i in range(len(trajectory)):
        pos = trajectory[i]
        x_pred.append(pos[0])
        y_pred.append(pos[1])  
        i += 100  
    # plt.plot(x, y, 'filter prediction')

    # ground_truth = load_data('./ds1_Groundtruth.dat', 3, 0, [0,4,5])
    x_gt = []
    y_gt = []
    for i in range(23000):
        x_gt.append(ground_truth[i][1])
        y_gt.append(ground_truth[i][2])

    fig = plt.figure()
    ax = plt.subplot(111)
    # ax.plot(x, y, label='odometry')
    ax.plot(x_pred, y_pred, label='filter prediction')
    ax.plot(x_gt, y_gt, label='ground_truth')
    plt.title('Increase likelihood noise')
    ax.legend()
    
    plt.savefig('./results/predict_trajectory_Q9_increase_likelihood_noise.png')    


def generate_barcode_dict(barcode_data):
    barcode_dict = {}
    for i in range(5,len(barcode_data)):
        barcode_dict[barcode_data[i][1]] = barcode_data[i][0]
    return barcode_dict

def generate_landmark_gt_dict(landmark_gt_data):
    landmark_gt_dict = {}
    for i in range(len(landmark_gt_data)):
        landmark_gt_dict[landmark_gt_data[i][0]] = [landmark_gt_data[i][1], landmark_gt_data[i][2]]
    return landmark_gt_dict

def whether_landmark(barcode):
    barcodes = [63,25, 45, 16, 61, 36, 18, 9, 72, 70, 81, 54, 27, 7, 90]
    return barcode in barcodes