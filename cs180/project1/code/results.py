from scaler_utils import *
from utils import *
import os
import json
import time
from collections import defaultdict
losses = {"l2Loss":l2Loss,"NCCLoss":NCCLoss}
scales= {"singleScale":singleScale,"multiScale":multiScale}
PATH="CS180_fa2026_proj1_data"
images = os.listdir(PATH)
dicts= {}
for scale,scaleFunc in scales.items():
    for loss,lossFunc in losses.items():
        for image in images:
            name,extension  = image.split(".")

            
            if (scale=="singleScale" and extension=="tif") or extension=="py" or image==".DS_Store":
                continue
            print(scale)
            outFp = f"{scale}/{loss}/{name}.jpg"
            fp = PATH+"/"+name+"."+extension
            print(fp)
            Start = time.time()
            (dgx,dgy),(drx,dry) = scaleFunc(fp,outFp,lossFunc,normalize_channel)
            End = time.time()
            dicts[str((scale,name,loss))] = {'r':[drx,dry],'g':[dgx,dgy],'time':End-Start}
            # dicts[scale][name][loss]['r']=[drx,dry]
            # dicts[scale][name][loss]['g']=[dgx,dgy]
json.dump(dicts,open('metadata.json','w'))





            

