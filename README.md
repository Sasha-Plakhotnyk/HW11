#Home work11

#import torch
#import cv2
#import numpy as np
#from PIL 
#import Image


resnet = models.resnet50(pretrained=True)
vgg = models.vgg16(pretrained=True)

# Переключение в режим оценки
resnet.eval()
vgg.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_url = "https://example.com/sample_image.jpg"  
image_path = "input_image.jpg"
import requests

response = requests.get(image_url)
with open(image_path, 'wb') as f:
    f.write(response.content)


original_image = Image.open(image_path).convert("RGB")
input_tensor = transform(original_image).unsqueeze(0)

with torch.no_grad():
    resnet_output = resnet(input_tensor)
    vgg_output = vgg(input_tensor)


categories = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json").json()
resnet_label = categories[resnet_output.argmax().item()]
vgg_label = categories[vgg_output.argmax().item()]

print(f"ResNet detected: {resnet_label}")
print(f"VGG detected: {vgg_label}")


image_cv = cv2.imread(image_path)
gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

# Применение детектора Canny
edges = cv2.Canny(gray, 100, 200)


contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50 and h > 50:  
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)


overlay = cv2.imread("overlay_image.jpg")
overlay = cv2.resize(overlay, (50, 50))  # Рзмір маски


for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50 and h > 50:
        roi = image_cv[y:y + h, x:x + w]
        overlay_resized = cv2.resize(overlay, (w, h))
        image_cv[y:y + h, x:x + w] = cv2.addWeighted(roi, 0.5, overlay_resized, 0.5, 0)


# Результат та збреження
output_path = "output_image.jpg"
cv2.imwrite(output_path, image_cv)

cv2.imshow("Result", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
