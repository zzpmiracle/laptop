import os, random, shutil

def random_cutFile(srcPath,dstPath,numfiles):
    name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
    random_name_list=list(random.sample(name_list,numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.move(oldname,oldname.replace(srcPath, dstPath))


root_path = 'D:\python\dataset\Dog_Cat'

test_nums = 250
srcPath=os.path.join(root_path,'train')
dstPath =os.path.join(root_path,'test')
random_cutFile(srcPath,dstPath,2500)
