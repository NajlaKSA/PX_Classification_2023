import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns

color = sns.color_palette("Blues", as_cmap=True, )

labels = ['Examination','Patient_Journey','Quality_Care','Treatment','Diagnosis','Medication_Vaccination','Safety_Inceidents','Skills_conducts','Administrative_Policies_procedures','Environment','Safety_Security','Finance_Billing','Staffing','Resources',
'Medical_Records','Access','Patient_Disposition','Delays',
'Referrals','Patient-Staff_Communication','Incorrect_Information','Emotional_Support','Assault_Harassment','Confidentiality',
'Consent']




gru = [[[2837,37]
,[10, 2]],
[[2875, 1]
,[10, 0]],
[[2229 ,258]
,[206 ,193]],
[[2778, 9]
,[94, 5]],
[[2867, 2]
,[15, 2]],
[[2718,53]
,[28,87]],
[[2784,39]
,[59, 4]],
[[2842,28]
,[13, 3]],
[[2750,61]
,[54,21]],
[[1508 ,132]
,[220,1026]],
[[2845, 4]
,[34, 3]],
[[2878, 1]
,[ 7, 0]],
[[2760,79]
,[15,32]],
[[2682,91]
,[70,43]],
[[2858, 5]
,[19, 4]],
[[2784,24]
,[30,48]],
[[2884, 0]
,[ 2, 0]],
[[2525 ,122]
,[68 ,171]],
[[2878, 0]
,[ 8, 0]],
[[2781 ,104]
,[ 1, 0]],
[[2881, 5]
,[ 0, 0]],
[[2846,32]
,[ 7, 1]],
[[2858,27]
,[ 1, 0]],
[[2865, 5]
,[12, 4]],
[[2886, 0]
,[ 0, 0]]]
"""
#--------------------

grut= [[[2854,20]
,[ 12, 0]],
[[2876, 0]
,[ 10, 0]],
[[2267 , 220]
,[225 , 174]],
[[2787, 0]
,[ 99, 0]],
[[2869, 0]
,[ 17, 0]],
[[2729,42]
,[ 46,69]],
[[2821, 2]
,[ 63, 0]],
[[2869, 1]
,[ 16, 0]],
[[2785,26]
,[ 59,16]],
[[1535 , 105]
,[219, 1027]],
[[2849, 0]
,[ 37, 0]],
[[2879, 0]
,[  7, 0]],
[[2808,31]
,[ 23,24]],
[[2751,22]
,[ 95,18]],
[[2863, 0]
,[ 23, 0]],
[[2777,31]
,[ 29,49]],
[[2884, 0]
,[  2, 0]],
[[2574,73]
,[ 88 , 151]],
[[2878, 0]
,[  8, 0]],
[[2874,11]
,[  1, 0]],
[[2886, 0]
,[  0, 0]],
[[2878, 0]
,[  8, 0]],
[[2881, 4]
,[  1, 0]],
[[2870, 0]
,[ 16, 0]],
[[2886, 0]
,[  0, 0]]]



lstm = [[[2852,22],[  11, 1]],
[[2876, 0],[  10, 0]],
[[2278 ,209],[ 207 , 192]],
[[2787, 0],[  99, 0]],
[[2869, 0],[  17, 0]],
[[2728,43],[  38,77]],
[[2823, 0],[  63, 0]],
[[2867, 3],[  16, 0]],
[[2779,32],[  58,17]],
[[1513 , 127],[ 216, 1030]],
[[2849, 0],[  37, 0]],
[[2879, 0],[7, 0]],
[[2812,27],[  22,25]],
[[2757,16],[ 104, 9]],
[[2863, 0],[  23, 0]],
[[2782,26],[  31,47]],
[[2884, 0],[2, 0]],
[[2546 , 101],[  65  ,174]],
[[2878, 0],[8, 0]],
[[2840,45],[1, 0]],
[[2886, 0],[0, 0]],
[[2878, 0],[8, 0]],
[[2870,15],[1, 0]],
[[2870, 0],[ 16, 0]],
[[2886, 0],[0, 0]]]


lstmt = [[[2864,10],[ 11, 1]],
[[2876, 0],[ 10, 0]],
[[2146 , 341],[152 , 247]],
[[2787, 0],[ 99, 0]],
[[2869, 0],[ 17, 0]],
[[2716,55],[ 34,81]],
[[2810,13],[ 62, 1]],
[[2864, 6],[ 12, 4]],
[[2804, 7],[ 68, 7]],
[[1473,  167],[143, 1103]],
[[2848, 1],[ 36, 1]],
[[2879, 0],[  7, 0]],
[[2805,34],[ 32,15]],
[[2608 , 165],[ 68,45]],
[[2863, 0],[ 23, 0]],
[[2779,29],[ 32,46]],
[[2884, 0],[  2, 0]],
[[2479 , 168],[ 50 , 189]],
[[2878, 0],[  8, 0]],
[[2879, 6],[  1, 0]],
[[2886, 0],[  0, 0]],
[[2878, 0],[  8, 0]],
[[2875,10],[  1, 0]],
[[2870, 0],[ 16, 0]],
[[2886, 0],[  0, 0]]]



"""



# ------------------------------------------------------------------------------------------------------------
def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=10):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, cmap=color)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)

def print_cf(cf):
    fig, ax = plt.subplots(5,5, figsize=(12, 10))
        
    for axes, cfs_matrix, label in zip(ax.flatten(), cf, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["0", "1"])

    fig.tight_layout()
    plt.show()

print_cf(gru)
#print_cf(grut)
#print_cf(lstm)
#print_cf(lstmt)
