import os
import shutil

for i in sorted(os.listdir('obj')):
    if '.txt' in i:
        print(i)
        with open('obj/'+i,'r')  as f:
            
            r = f.readlines()
            changed = []
            for ix in r:
                
                ix = list(ix)
                ix[0]=str(int(ix[0])+1)
                ix="".join(ix)
                changed.append(ix)
        # print(changed)
        with open('obj/'+i,'w') as fe:
            fe.writelines(changed)
            