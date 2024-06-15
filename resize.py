from PIL import Image
import os


fin = r"D:\liuzhihuan\HFANet\images\LEVIR_CD\test\B"     # 输入图像所在路径
fout = r"D:\liuzhihuan\HFANet\images\LEVIR_CD\test\B1"    # 输出图像的路径
for file in os.listdir(fin):
    file_fullname = fin + '/' +file
    print(file_fullname)                            # 所操作图片的名称可视化
    img = Image.open(file_fullname)
    im_resized = img.resize((256, 256))             # resize至所需大小
    out_path = fout + '/' + file
    im_resized.save(out_path)                       # 保存图像

