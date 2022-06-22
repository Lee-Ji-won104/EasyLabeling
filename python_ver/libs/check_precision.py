TP=0
TN=0
FP=0
FN=0 


def calculate_MBTI(class_id, fileName2):

    label=class_id
    global TP
    global TN
    global FP
    global FN   

    if 'car' in fileName2:
        if label == 'car' or label == 'truck' or label =='bus':
            TP+=1
        else:
            FN+=1
        
    elif 'conv'==fileName2:
        if label == 'car' or label == 'truck' or label =='bus':
            FP+=1
        else:
            TN+=1

def calculate_All():

    global TP
    global TN
    global FP
    global FN  
    
    ALL_list=[]
    accuracy=cal_Accuracy(TP, TN, FP, FN)
    recall=cal_Recall(TP,FN)
    precision=cal_Precision(TP,FP)
    F1=cal_F1(precision, recall)

    ALL_list.append(accuracy)
    ALL_list.append(recall)
    ALL_list.append(precision)
    ALL_list.append(F1)

    return ALL_list

def cal_Accuracy(TP,TN,FP,FN):

    return (TP+TN)/(TP+TN+FP+FN)*100

def cal_Recall(TP, FN):
    return TP/(TP+FN)*100

def cal_Precision(TP,FP):
    return TP/(TP+FP)*100

def cal_F1(precision, recall):
    return (2*precision*recall)/(precision+recall)