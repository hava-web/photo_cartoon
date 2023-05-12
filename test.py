import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess
from flask import Flask, request
import requests
import base64
import json
import shutil

app = Flask(__name__)
api_key = "7f9fc33be9d03f6775f31d9af2af3858"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 OPR/96.0.0.0'}
# parser = argparse.ArgumentParser()
# parser.add_argument('--photo_path', type=str, help='input photo path')
# parser.add_argument('--save_path', type=str, help='cartoon save path')
# args = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

def download_image(url, filename):
    response = requests.get(url, stream=True,headers=headers)
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return filename


def upload_image_to_imgbb(image_path, api_key):
    # Tải dữ liệu ảnh
    with open(image_path, "rb") as file:
        payload = {
            "key": api_key,
            "image": base64.b64encode(file.read()),
        }

    # Gửi yêu cầu POST tải lên ảnh đến API của ImgBB
    response = requests.post("https://api.imgbb.com/1/upload", payload)

    # Trích xuất đường dẫn trực tiếp đến ảnh từ JSON response
    json_data = json.loads(response.text)
    # print(json_data)
    direct_link = json_data["data"]["url"]
    # Trả về đường dẫn trực tiếp đến ảnh
    return direct_link

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        assert os.path.exists('./models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon

@app.route("/image", methods=["GET"])
def main():
    link = dict()
    link_img = request.headers.get('link-img')
    filename = 'input_img.png'
    img = download_image(link_img,filename)
    img_read = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(img_read)
    cv2.imwrite('my_picture.png',cartoon)
    direct_link = upload_image_to_imgbb("my_picture.png",api_key)
    link['link'] = direct_link
    print('Cartoon portrait has been saved successfully!')
    return json.dumps(link)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
    # img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
    # c2p = Photo2Cartoon()
    # cartoon = c2p.inference(img)
    # if cartoon is not None:
    #     cv2.imwrite(args.save_path, cartoon)
    #     print('Cartoon portrait has been saved successfully!')
