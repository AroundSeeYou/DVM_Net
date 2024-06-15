'''
import os
#
root_path = r'D:\liuzhihuan\HFANet\images\LEVIR-CD\train\label'  # 要修改的图像所在的文件夹路径

filelist = os.listdir(root_path)  # 遍历文件夹
i = 0
for item in filelist:

    if item.endswith('.png'):
        src = os.path.join(os.path.abspath(root_path), item)  # 原本的名称

        src1 = src.split('_')
        print(src1[1])
        ''''''
        dst = os.path.join(os.path.abspath(root_path), src1[1])  # 这里我把格式统一改成了 .jpg

        try:
            os.rename(src, dst)  # 意思是将 src 替换为 dst
            i += 1
            print('rename from %s to %s' % (src, dst))
        except:
            continue


print('ending...')
'''
from PIL import Image
import os
import cv2






fin = r"D:\liuzhihuan\HFANet\images\LEVIR_CD\train\label"     # 输入图像所在路径
fout = r"D:\liuzhihuan\HFANet\images\LEVIR_CD\train\label1"     # 输出图像的路径

for file in os.listdir(fin):
    file_fullname = fin + '/' +file
    print(file_fullname)                            # 所操作图片的名称可视化
    # img = Image.open(file_fullname)
    img = cv2.imread(file_fullname)
    up_points = (512, 512)
    resized_up = cv2.resize(img, up_points, interpolation=cv2.INTER_LINEAR)
    # im_resized = img.resize((256, 256))             # resize至所需大小

    out_path = fout + '/' + file
    # resized_up.save(out_path)                       # 保存图像
    cv2.imwrite(out_path, resized_up)

