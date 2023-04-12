import torch
from torch.nn.parameter import Parameter
from PIL import Image
from torchvision.models import resnet34,resnet50,resnet152,ResNet34_Weights,ResNet50_Weights,ResNet152_Weights
#import numpy as np
#import time
import json
from pathlib import Path
from nltk.tokenize import word_tokenize
import pickle
#import torch.multiprocessing as mp

#cross_class_activdic={}
echimgdict={}
masknext=0
memo=[]
pklmemo=[]

with open('C:/Users/lenovo/repos/my/imagenet-simple-labels/imagenet-simple-labels.json') as f:
    labels = json.load(f)
def class_id_to_label(i):
    return labels[i]

flattenedmdlist=[]
mpltllist=[]
mklist=[]
flag=0
restoreflag=0
prevalue=Parameter(torch.Tensor([0]))

def flatten():
    global flattenedmdlist
    flattenedmdlist=[]
    for k in rsn._modules.keys():
        kk=list(rsn._modules[k].children())
        lenthlvl1=len(kk)
        if lenthlvl1==0:
            flattenedmdlist.append('rsn'+'.'+k)
            continue
        else:
            for l in range(lenthlvl1):
                for m in kk[l]._modules.keys():
                    if m:
                        lm=list(kk[l]._modules[m].children())
                        lenthlvl2=len(lm)
                        if lenthlvl2==0:
                            flattenedmdlist.append('rsn'+'.'+k+'['+str(l)+']'+'.'+m)
                            continue
                        else:
                            for n in range(lenthlvl2):
                                       no=list(lm[n].children())
                                       lenthlvl3=len(no)
                                       if lenthlvl3==0:
                                           flattenedmdlist.append('rsn'+'.'+k+'['+str(l)+']'+'.'+m+'['+str(n)+']'+'.'+lm[n]._get_name())
                                           continue
                                       else:
                                           print('Unsupport Deepth!')
                                           continue
                    else:
                        flattenedmdlist.append('rsn'+'.'+k+'['+str(l)+']')
                        continue

def inspec(numh,nmh):
    def hk(module,args,output):
        outputanalysis=output
        #print()
        outputanalysis=outputanalysis.squeeze(0)
        max=outputanalysis.max()
        min=outputanalysis.min()
        outputanalysis=(outputanalysis-min)*255/(max-min)
        #hmax=outputanalysis.max()
        #hmin=outputanalysis.min()
        #mid=(hmax-hmin)/2
        #print(nmh,module._get_name(),output.shape,outputanalysis.shape,max,"Max:Min",min)
        f=0
        ct=0
        masklist=[]
        global flag
        for i in outputanalysis:
            item=torch.Tensor([o for o in i.reshape(-1) if o!=0])
            #num=torch.count_nonzero(item)
            varmean=torch.var_mean(item)
            if (varmean[0] > 16 or varmean[1] > 8) or ('nan' in str(varmean[0]) and varmean[1] > 20):
            #if (varmean[0] > 5 or varmean[1] > 1) or ('nan' in str(varmean[0]) and varmean[1] > 5):
                masklist.append(1)
                #img=Image.fromarray(i.detach().numpy()).convert("L")
                #name="E:/inspect/"+str(f)+".jpg"
                ct+=1
                #img.save(name)
            else:
                masklist.append(0)
                #img=Image.fromarray(i.detach().numpy()).convert("L")
                #name="E:/deactive/"+str(f)+".jpg"
                #img.save(name)
            f+=1
        #print("###",ct,"Activing Layers","--- saved layers:",outputanalysis.shape[0]-ct )
        global masknext
        global mklist
        mklist.append(masklist)
        masknext = torch.Tensor(masklist)
        if flag < len(mpltllist):
                if hasattr(eval(mpltllist[flag]),'weight'):
                        global prevalue
                        global restoreflag
                        if torch.any(prevalue) and restoreflag:
                            if flag != 0:
                                eval(mpltllist[flag-1]).weight=prevalue
                                #print(mpltllist[flag-1],'layer changes restored!!!')
                            else:
                                rsn.fc.weight=prevalue
                                #print('rsn.fc','layer changes restored!!!')
                        prevalue = Parameter(eval(mpltllist[flag]).weight)
                        #print(prevaule.shape,masknext.shape)
                        if 'fc' in mpltllist[flag]:
                            eval(mpltllist[flag]).weight = Parameter(torch.mul(eval(mpltllist[flag]).weight.unsqueeze(2).unsqueeze(3),torch.unsqueeze(torch.unsqueeze(masknext,1),1)).squeeze(3).squeeze(2))
                            restoreflag=1
                        else:
                            eval(mpltllist[flag]).weight = Parameter(torch.mul(eval(mpltllist[flag]).weight,torch.unsqueeze(torch.unsqueeze(masknext,1),1)))
                            echimgdict[mpltllist[flag]]=masknext
                        aftervalue = eval(mpltllist[flag]).weight
                        if torch.any(torch.not_equal(prevalue,aftervalue)):
                                #print(flag,mpltllist[flag],prevalue.shape,aftervalue.shape,'Changed!!!!!!!!!!!!!!!!!!!!!!')
                                restoreflag=1
                        else:
                                #print('Nothing happened.')
                                restoreflag=0
                else:
                    #print("Next Layer no parameters!!!")
                    restoreflag=0
                flag+=1
    return hk

#weights = ResNet34_Weights.DEFAULT
weights = ResNet50_Weights.DEFAULT
#weights = ResNet152_Weights.DEFAULT
#weights.get_state_dict(progress=True)
#rsn = resnet34(weights=weights)
rsn = resnet50(weights=weights)
#rsn = resnet152(weights=weights)
flatten()
for (num,nm) in enumerate(flattenedmdlist):
    if 'relu' in nm or 'avgpool' in nm or 'fc' in nm:
    #if 'relu' in nm or 'avgpool' in nm:
        eval(nm).register_forward_hook(inspec(num,nm))
    #elif 'bn' in nm or 'BatchNorm' in nm:
    elif 'bn' in nm:
        global mpltllist
        if 'relu' in flattenedmdlist[num+1]:
            if 'maxpool' in flattenedmdlist[num+2]:
                mpltllist.append(flattenedmdlist[num+3])
            elif 'BatchNorm' in flattenedmdlist[num+3]:
                mpltllist.append(flattenedmdlist[num+4])
            else:
                mpltllist.append(flattenedmdlist[num+2])
        else:
            mpltllist.append(flattenedmdlist[num+1])
        continue
    else:
        continue
mpltllist.append('rsn.fc')

rsn.eval()

preprocess = weights.transforms()

imgpath=Path(r'E:\BaiduNetdiskDownload\ILSVRC2012\train')

for eachdir in imgpath.iterdir():
    counter=0
    ANVLU={}
    VLU={}
    errorflag=0
    prevlabel=''
    labelflag=0
    mistkcounter=0
    for image in list(eachdir.glob('**/*.JPEG')):
        print('Image:',eachdir,image)
        img=Image.open(image)
        if 'L' ==  img.getbands()[0]:
            continue
        echimgdict={}
        masknext=0
        mklist=[]
        flag=0
        id=-1
        label=''
        T=torch.softmax(torch.Tensor(rsn(preprocess(img).unsqueeze(0))),1)
        for i in range(0,999):
            if T.max() == T.squeeze(0)[i]:
                id=i
                print(i,"||",class_id_to_label(i))
                label='_'.join([z for z in word_tokenize(class_id_to_label(i)) if z!=' '])
                if not prevlabel:
                    prevlabel=label
                else:
                    if prevlabel == label:
                        labelflag=0
                    else:
                        labelflag=1
                        prevlabel = label
                if label not in memo:
                    globals()[label]=[]
                    memo.append(label)
                elif label in pklmemo:
                    #print('Mistake happend!!!')
                    errorflag=1
                    mistkcounter+=1
                    break
                globals()[label].append(echimgdict)
#                cross_class_activdic[label]=eval(label)
                counter+=1
                errorflag=0
                break
            else:
                continue
        if counter%100 == 0 and not errorflag and not labelflag:
            for anlz in mpltllist:
                if 'layer' in anlz:
                    lenth=len(eval(label))
                    plenth=len(eval(prevlabel))
                    if lenth<20 and label != prevlabel and plenth>40:
                        label=prevlabel
                        lenth=plenth
                    VLU[anlz]=torch.zeros_like(eval(label)[0][anlz])
                    for a in range(lenth):
                        VLU[anlz]+=eval(label)[a][anlz]
                    VLU[anlz]=torch.floor(VLU[anlz]/lenth)
                    print('XXXX',anlz,eval(label)[a][anlz].shape,int(torch.count_nonzero(VLU[anlz])))
                else:
                    if label not in pklmemo:
                        #print('Pickle and Save!')
                        outputfilename=str(id)+'_'+label+'_'+str(mistkcounter)+'.pkl'
                        output = open(outputfilename, 'wb')
                        pickle.dump(eval(label), output)
                        output.close()
                        del globals()[label]

                        ANVLU[label]=VLU
                        outputfilename=str(id)+'_value_'+label+'_'+str(mistkcounter)+'.pkl'
                        output = open(outputfilename, 'wb')
                        pickle.dump(ANVLU, output)
                        output.close()
                        del ANVLU
                        pklmemo.append(label)
#                    outputfilenameraw=str(id)+'_'+label+'_raw'+'.pkl'
#                    outputraw = open(outputfilenameraw, 'wb')
#                    pickle.dump(cross_class_activdic, outputraw)
#                    outputraw.close()
                    for name in memo:
                        if name not in pklmemo and len(name)<40:
                            memo.remove(name)
                    break
            break
        elif mistkcounter>10:
            #print('Go to Next!!!')
            break
        elif labelflag:
            counter-=1
            continue
        #break


#pkl_file = open('myfile.pkl', 'rb')
#mydict2 = pickle.load(pkl_file)
#pkl_file.close()

        
