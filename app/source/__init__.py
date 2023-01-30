# requirements 설치 : pip install -r requirements.txt
import torch
import cv2
import matplotlib.pyplot as plt

# @webius - import 추가
import numpy
import json
import io
from PIL import Image
import base64

# @webius - app 패키지에서 import 하기 위해 경로 수정
from app.source.code.src.Models import Unet

# 한글 깨짐 오류 해결
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
# Flask 전달 시 사용할 attribute names
areaLabels = ['breakage', 'crushed', 'scratch', 'seperated']
models = []

n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for label in labels:
  # app 패키지에서 로드하기 위해 경로 수정
  model_path = f'flask_car_accident/app/source/models/[DAMAGE][{label}]Unet.pt'

  model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
  model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
  model.eval()

  models.append(model)

from enlighten_inference import EnlightenOnnxModel
# pip install git+https://github.com/arsenyinfo/EnlightenGAN-inference

# Flask에서 읽을 수 있도록 함수 형태로 구성
def getPrice(file=None):
  global device

  response = {}

  if file is not None:
    # 저장 과정을 거치지 않고 바로 로드
    img = cv2.imdecode(numpy.fromstring(file.read(), numpy.uint8), cv2.IMREAD_COLOR)
    print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    model_light = EnlightenOnnxModel()
    img = model_light.predict(img)

  plt.figure(figsize=(8, 8))
  plt.imshow(img)

  img_input = img / 255.
  img_input = img_input.transpose([2, 0, 1])
  img_input = torch.tensor(img_input).float().to(device)
  img_input = img_input.unsqueeze(0)

  fig, ax = plt.subplots(1, 5, figsize=(24, 10))

  ax[0].imshow(img)
  ax[0].axis('off')
  # subplot 이름 변경
  ax[0].set_title('원본', fontsize=30)
  ax[1].set_title('파손', fontsize=30)
  ax[2].set_title('찌그러짐', fontsize=30)
  ax[3].set_title('스크래치', fontsize=30)
  ax[4].set_title('이격', fontsize=30)

  outputs = []

  for i, model in enumerate(models):
    output = model(img_input)

    img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
    img_output = img_output.transpose([1, 2, 0])

    outputs.append(img_output)

    # ax[i+1].set_title(labels[i])
    # ax[i+1].imshow(img_output, cmap='jet')
    # plot 사진 겹쳐서 보이게 하기
    ax[i + 1].imshow(img.astype('uint8'), alpha=0.9)
    ax[i + 1].imshow(img_output, cmap='jet', alpha=0.6)
    ax[i+1].axis('off')

  fig.set_tight_layout(True)

  # 이미지 전달
  image = io.BytesIO()
  plt.savefig(image, format='jpeg')
  response['image'] = base64.b64encode(image.getvalue()).decode('utf-8').replace('\n', '')

  for i, label in enumerate(labels):
    print(f'{label}: {outputs[i].sum()}')

  price_table = [
    120, # Breakage_3 / 파손 200
    90, # Crushed_2 / 찌그러짐 150
    60,  # Scratch_0 / 스크래치 100
    120, # Seperated_1 / 이격 200
  ]

  total = 0

  for i, price in enumerate(price_table):
    area = outputs[i].sum()
    total += area * price

    print(f'{labels[i]}:\t영역: {area}\t가격:{area * price}원')

  print(f'고객님, 총 수리비는 {total}원 입니다!')

  # 총 수리비 전달
  response['total'] = int(total)


  # 전체 면적 계산
  # app 패키지에서 로드하기 위해 경로 수정
  weight_path = 'flask_car_accident/app/source/models/[PART]Unet.pt'

  n_classes = 16
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
  model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
  model.eval()

  img_input = img / 255.
  img_input = img_input.transpose([2, 0, 1])
  img_input = torch.tensor(img_input).float().to(device)
  img_input = img_input.unsqueeze(0)

  output = model(img_input)

  img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
  img_output = img_output.transpose([1, 2, 0])

  area_sum = img_output.sum()

  # 각각 손상부위별 면적
  area_breakage = outputs[0].sum()
  area_crushed = outputs[1].sum()
  area_scratch = outputs[2].sum()
  area_seperated = outputs[3].sum()
  print(area_sum, area_breakage, area_crushed, area_scratch, area_seperated)

  # 각 손상 부위별 면적 전달
  response['area'] = {}
  for i, label in enumerate(areaLabels):
    size = int(outputs[i].sum())
    price = price_table[i]

    response['area'][label] = {
      'size': size,
      'price': price * size,
    }

  severity = (area_breakage*3.0 + area_crushed*2.0 + area_seperated*1.2 + area_scratch*1.0) * 100 / (3*area_sum)
  # severity


  if 0 <= severity < 11:
    grade = 4
  elif  severity < 41:
    grade = 3
  elif  severity < 81:
    grade = 2
  else:
    grade = 1

  print('손상심각도 :', grade, '등급')

  # 등급 전달
  response['grade'] = grade

  return json.dumps(response, separators=(',', ':'))
