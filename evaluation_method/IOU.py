# encoding: utf-8

def compute_iou(rec1, rec2):
    """
    param:
    rec1: (x0, y1, w, h)
    rec2: (x0, y1, w, h)
    x0, y0: the upper left point of rec.
    w, h:  the length and width of rec.
    """
    left_x = max(rec1[0], rec2[0])
    left_y = max(rec1[1], rec2[1])
    right_x = min(rec1[0] + rec1[2], rec2[0] + rec2[2])
    right_y = min(rec1[1] + rec1[3], rec2[1] + rec2[3])
    if left_x >= right_x or left_y >= right_y:
        return 0
    else:
        S_mid = (right_y - left_y)*(right_x - left_x)
        S_total = (rec1[2]*rec1[3]) + (rec2[2]*rec2[3]) - S_mid
        return S_mid/S_total

rec1 = [2,3,8,9]
rec2 = [12,5,8,19]
IOU = compute_iou(rec1, rec2)
print ("Case1 IOU: %f" % IOU)

rec1 = [2,2,2,2]
rec2 = [3,3,2,2]
IOU = compute_iou(rec1, rec2)
print ("Case2 IOU: %f" % IOU)
