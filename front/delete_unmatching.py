import glob,os

files_results = glob.glob('D:\\SIH fabric-test\\dataset\\results\\*.png')
files_defects = glob.glob('D:\\SIH fabric-test\\dataset\\original\\Defect_images\\*.png')

a = [f[-15:] for f in files_defects]
b = [f[-15:] for f in files_results]

for i in a:
    if(i in b):
        continue
    else:
        os.remove('D:\\SIH fabric-test\\dataset\\original\\Defect_images\\'+i)

