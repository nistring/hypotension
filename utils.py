from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

u_lim = 150
l_lim = 40
delta_lim = 50

def rolling_window(array, window_size, freq):
    array = array[:int(len(array) // freq * freq)]
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],freq)]

def remove_na(mbps):
    # a-line 연결 전 샘플들을 제거
    with np.errstate(invalid="ignore"):
        mbps[mbps < 40] = np.nan

    # 처음과 마지막의 결측값을 제거
    case_valid_mask = ~np.isnan(mbps)
    mbps = mbps[
        (np.cumsum(case_valid_mask) != 0)
        & (np.cumsum(case_valid_mask[::-1])[::-1] != 0)
    ]

    # 중간 결측값을 직전값으로 대체
    mbps = pd.DataFrame(mbps).ffill().values.flatten().astype("float32")
    
    return mbps
        
def preprocessing(mbps, in_horizon, out_horizon):
    if len(mbps) < 1800:
        return np.array([]), np.array([])
    
    mbps = remove_na(mbps)
    interval = in_horizon + out_horizon
    
    if len(mbps) < interval:
        return np.array([]), np.array([])
    
    x = rolling_window(mbps, interval, in_horizon)
    valid_mask = np.logical_not(np.any(x > u_lim, axis=1) * np.any(np.abs(np.diff(x, axis=1)) > delta_lim, axis=1) * np.any(x < l_lim, axis=1))
    x, y = x[valid_mask, :in_horizon], x[valid_mask, in_horizon:]
    y = (np.nanmax(y, axis=1) < 65).astype("float32")
    
    return x, y


def vis(test_y_valid, test_y_pred):
    precision, recall, thmbps = precision_recall_curve(test_y_valid, test_y_pred)
    auprc = auc(recall, precision)

    fpr, tpr, thmbps = roc_curve(test_y_valid, test_y_pred)
    auroc = auc(fpr, tpr)

    thval = 0.5
    f1 = f1_score(test_y_valid, test_y_pred > thval)
    acc = accuracy_score(test_y_valid, test_y_pred > thval)
    tn, fp, fn, tp = confusion_matrix(test_y_valid, test_y_pred > thval).ravel()

    testres = 'auroc={:.3f}, auprc={:.3f} acc={:.3f}, F1={:.3f}, PPV={:.1f}, NPV={:.1f}, TN={}, fp={}, fn={}, TP={}'.format(auroc, auprc, acc, f1, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp)
    print(testres)

    # auroc curve
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('auroc.png')
    plt.close()

    # auprc curve
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('auprc.png')
    plt.close()

    # Predict step to be implemented

    # # 각 case 그림
    # for caseid in caseids_test:
    #     case_mask = (y_caseid == caseid)
    #     case_len = np.sum(case_mask)
    #     if case_len == 0:
    #         continue

    #     # case 내의 x, y, valid_mask 를 만든다
    #     case_x = x[case_mask]
    #     case_y = y[case_mask]
    #     case_valid_mask = valid_mask[case_mask]
        
    #     # case 에러를 구하고 출력
    #     case_predy = model.predict(case_x).flatten()
    #     case_rmse = np.nanmean((case_y - case_predy) ** 2) ** 0.5
    #     print('{}\t{}\t'.format(caseid, case_rmse))

    #     # 그림 생성
    #     plt.figure(figsize=(20, 4))
    #     plt.xlim([0, case_len + MINUTES_AHEAD * 30])
    #     t = np.arange(0, case_len)

    #     # 저혈압 상태일 때를 붉은 반투명 배경으로
    #     ax1 = plt.gca()
    #     for i in range(len(case_y)):
    #         if case_y[i]:
    #             ax1.axvspan(i + MINUTES_AHEAD * 30, i + MINUTES_AHEAD * 30 + 1, color='r', alpha=0.1, lw=0)

    #     # 65 mmHg 가로선
    #     ax1.axhline(y=65, color='r', alpha=0.5)
    #     ax1.plot(t + 10, case_x[:,-1], color='r')
    #     ax1.set_ylim([0, 150])

    #     ax2 = ax1.twinx()
        
    #     # valid 한 샘플만 그린다
    #     case_predy[~case_valid_mask] = np.nan
    #     ax2.plot(t, case_predy)
    #     ax2.set_ylim([0, 1])
        
    #     # 그림 저장
    #     plt.show()