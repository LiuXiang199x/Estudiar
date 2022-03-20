from PIL import Image
def make_416_image(image_path):
    img=Image.open(image_path)
    w,h=img.size[0],img.size[1]
    temp=max(w,h)
    mask=Image.new(mode='RGB',size=(temp,temp),color=(0,0,0))
    mask.paste(img,(0,0))
    return mask


if __name__ == '__main__':
    mask=make_416_image('/home/marco/Estudiar/DeepLearning/Project/yolo-v3/dataset/dog/1.jpg')
    mask.show()