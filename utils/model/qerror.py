import numpy as np


def compute_percentile_q_error(preds, labels, cuda, do_scale, percentile, min_val=1.0):
    """ compute q-error

    :param preds: prediction cardniailty (type: numpy.ndarray | torch.Tensor)
    :param labels: ground truth (type: numpy.ndarray | torch.Tensor)
    :param cuda: whether data is in cuda, if True, move data to cpu (type: bool)
    :param do_scale: whether to scale data to original value (type: bool)
    :param percentile: percentile of q-error (type: int)
    :param min_val: make sure the value is larger than min_val (type: float)
    :return: q-error (type: float)
    """
    if cuda:
        preds = preds.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
    if do_scale:
        preds = np.exp(preds - 1) - 1
        labels = np.exp(labels - 1) - 1
    preds = preds.astype(np.float64) + min_val
    labels = labels.astype(np.float64) + min_val

    q_errors = np.maximum(labels / preds, preds / labels)
    q_errors = np.nan_to_num(q_errors, nan=np.inf)
    median_q = np.percentile(q_errors, percentile)
    return median_q

def get_qerror(pred, label, cuda, do_scale, percentile_list):
    """ get q-error list

    :param pred: prediction cardniailty (type: numpy.ndarray | torch.Tensor)
    :param label: ground truth (type: numpy.ndarray | torch.Tensor)
    :param cuda: whether data is in cuda, if True, move data to cpu (type: bool)
    :param do_scale: whether to scale data to original value (type: bool)
    :param percentile_list: percentile of q-error (type: list)
    :return: q-error (type: list)
    """
    q_list = []
    for percentile in percentile_list:
        assert percentile >= 0 and percentile <= 100
        qerror = compute_percentile_q_error(pred, label, cuda=cuda, do_scale=do_scale, percentile=percentile)
        q_list.append(round(qerror, 4))
    return q_list

def interval_qerror(preds, labels, cuda, do_scale):
    """ get q-error in each interval

    :param preds: prediction cardniailty (type: numpy.ndarray | torch.Tensor)
    :param labels: ground truth (type: numpy.ndarray | torch.Tensor)
    :param cuda: whether data is in cuda, if True, move data to cpu (type: bool)
    :param do_scale: whether to scale data to original value (type: bool)
    :return: q-error in each interval (type: list)
    """
    if cuda:
        preds = preds.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
    if do_scale:
        preds = np.exp(preds - 1) - 1
        labels = np.exp(labels - 1) - 1

    interval = [10 ** i for i in range(int(np.log10(np.max(labels))) + 2)]
    results = []
    for i in range(len(interval) - 1):
        left_bound = interval[i]
        right_bound = interval[i + 1]

        interval_preds = preds[np.logical_and(labels >= left_bound, labels < right_bound)]
        interval_labels = labels[np.logical_and(labels >= left_bound, labels < right_bound)]

        if len(interval_preds) > 0:
            gt_count = np.sum(interval_preds > interval_labels)
            gt_ratio = gt_count / len(interval_preds)
            lt_count = np.sum(interval_preds < interval_labels)
            lt_ratio = lt_count / len(interval_preds)

            q_error = get_qerror(pred=interval_preds, label=interval_labels, cuda=False, do_scale=False, percentile_list=[30, 50, 80, 90, 95, 99])
        else:
            gt_count, gt_ratio, lt_count, lt_ratio, q_error = 0, 0, 0, 0, [0, 0, 0, 0, 0, 0]

        results.append((left_bound, right_bound, lt_count, lt_ratio, gt_count, gt_ratio, q_error))
    
    print("group_limits Num_of_underest pct_of_underest Num_of_overest pct_of_overest q_error_30% q_error_50% q_error_80% q_error_90% q_error_95% q_error_99%")
    for idx, group in enumerate(results):
        print(f'[1e{idx}, 1e{idx + 1})', end=' ')
        print(f'{group[2]:3.2f} ', f'{group[3]:.2f} ', f'{group[4]:3.2f} ', f'{group[5]:.2f} ', f'{group[6][0]:.2f} ', f'{group[6][1]:.2f} ', f'{group[6][2]:.2f} ', f'{group[6][3]:.2f} ', f'{group[6][4]:.2f} ', f'{group[6][5]:.2f}')
