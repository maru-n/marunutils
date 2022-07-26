from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import builtins
import numpy as np


def _plot_reassign_result(src_data, src_label, dst_data, dst_label, assign_map):
    plt.axes().set_aspect('equal')
    plt.plot(src_data[:,0], src_data[:,1], 'o', c='red', label='src')
    for (x,y), l in zip(src_data[:,:2], src_label):
        if l in assign_map.keys():
            plt.text(x+0.01,y-0.05, f'{l}=>{assign_map[l]}',c='red')
        else:
            plt.text(x+0.01,y-0.05, f'{l}',c='red')

    plt.plot(dst_data[:,0], dst_data[:,1], 'x', c='blue', label='dst')
    for (x,y), l in zip(dst_data, dst_label):
        plt.text(x+0.01,y+0.01,l,c='blue')

    plt.legend()
    plt.show()


def reassign(src_data, dst_data, src_label=None, dst_label=None, 
             metric='euclidean', dist_threshold=None,
             assign_new_label=True, new_label=None, new_label_increment=True,
             verbose=False, visualize_result=False):
    """reassign

    Generate reassign map to convert src_label to dst_label by nearest neighbor using the Hungarian algorithm based on the distance (metric) between src_data and dst_data.

    Args:
    src_data: source data (NxM)
    dst_data: destination data (LxM)
    src_label: labels of src_data (N)
    dst_label: labels of dst_data (L)
    metric: metric for cost matrix
    dist_threshold: maximum distance to connect the data points
    assign_new_label: assign new_label to all points which not assigned
    new_label: label to use if there is no point to assign
    new_label_increment: if True and new_label is int, new label will be increment

    Returns:
    ndarray: labels reassigned for source data (same length of src_data)
    """
    
    # print info if verbose is true
    def print(*args, **kwargs):
        if verbose:
            return builtins.print(*args, **kwargs)
    
    if src_label is None:
        src_label = np.arange(len(src_data))
    if dst_label is None:
        dst_label = np.arange(len(dst_data))
    
    assert src_data.shape[0] == len(src_label), f'shape of src_data {src_data.shape} is not match to length of src_label {len(src_label)}.'
    assert dst_data.shape[0] == len(dst_label), f'shape of dst_data {dst_data.shape} is not match to length of dst_label {len(dst_label)}.'
    
    # assert set(target) <= set(src_label), f'labels ({list(set(target)-set(src_label))}) in target is not included in src_label'

    print(f'{len(src_label)} points will be assigned to {len(dst_label)} points')
    
    cost = distance.cdist(src_data, dst_data, metric=metric)
    if verbose:
        print(f'max value of cost matrix: {cost.max()}')
        print(f'min value of cost matrix: {cost.max()}')
    
    exceed_threshold_tmp = cost.max() * 2
    if dist_threshold is not None:
        print(f'{np.sum(cost>dist_threshold)} points in {len(src_data)}x{len(dst_data)} is exceeded the threshold')
        # cost[cost>dist_threshold] = np.finfo(cost.dtype).max
        cost[cost>dist_threshold] = exceed_threshold_tmp
    
    src_idx, dst_idx = linear_sum_assignment(cost)

    if dist_threshold is not None:
        for i in range(len(src_idx)):
            # if cost[src_idx[i],dst_idx[i]] == np.finfo(cost.dtype).max:
            if cost[src_idx[i],dst_idx[i]] >= exceed_threshold_tmp:
                dst_idx[i] = -1

    cost_sum = cost[src_idx[dst_idx>=0], dst_idx[dst_idx>=0]].sum()
    print(f'total cost: {cost_sum}')
                        
    assign_map = {}
    new_label_num = 0
    for i, j in zip(src_idx, dst_idx):
        s = src_label[i]
        if j != -1:
            assign_map[s] = dst_label[j]
        # elif assign_new_label:
        #     assign_map[s] = new_label
        #     new_label_num += 1
        #     if type(new_label) is int and new_label_increment:
        #         new_label += 1
        # else:
        #     continue
    
    if assign_new_label:
        for s in src_label:
            if not s in assign_map.keys():
                assign_map[s] = new_label
                new_label_num += 1
                if type(new_label) is int and new_label_increment:
                    new_label += 1
    
    print(f'result of assignment')
    for a, b in assign_map.items():
        print(f'{a} => {b}')
    print(f'{len(assign_map)-new_label_num} points will be assigned to existing labels and {new_label_num} points will be assigned to new label in {len(src_data)} data points')

    if visualize_result:
        _plot_reassign_result(src_data, src_label, dst_data, dst_label, assign_map)
        
    return assign_map


if __name__ == "__main__":
    N = 5

    X = np.random.rand(N,2)
    Y = np.concatenate((X+np.random.normal(size=(N,2))*0.01, np.random.rand(N,2)), axis=0)
    np.random.shuffle(Y)
    X = np.concatenate((X, np.random.rand(N,2)), axis=0)

    X_label = np.random.choice(range(100), X.shape[0], replace = False)
    Y_label = np.random.choice(range(100), Y.shape[0], replace = False)

    assign_map = reassign(X, Y, X_label, Y_label, dist_threshold=0.1, new_label=1000, verbose=True, visualize_result=True)
    print(assign_map)

    # test
    # check by-direction
    m1 = reassign(X, Y, X_label, Y_label, dist_threshold=0.1, assign_new_label=False)
    m2 = reassign(Y, X, Y_label, X_label, dist_threshold=0.1, assign_new_label=False)
    for i,j in m1.items():
        assert i == m2[j]
    for i,j in m2.items():
        assert i == m1[j]
