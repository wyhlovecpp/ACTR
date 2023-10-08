from PIL import Image

# 打开并加载要拼接的三张图片
image1 = Image.open('eg1.png')
image2 = Image.open('eg2.png')
image3 = Image.open('eg8.png')

# 获取图片的宽度和高度
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size

# 计算拼接后图片的宽度和高度
width_total = width1 + width2 + width3
height_total = max(height1, height2, height3)

# 创建一个新的空白图片，大小为拼接后的尺寸
merged_image = Image.new('RGB', (width_total, height_total))

# 将三张图片按顺序拼接到新的图片上
merged_image.paste(image1, (0, 0))
merged_image.paste(image2, (width1, 0))
merged_image.paste(image3, (width1 + width2, 0))

# 保存拼接后的图片
merged_image.save('merged_image_3.jpg')