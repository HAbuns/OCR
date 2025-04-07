from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

text_det_model_path = "/yolo_models/detect/train2/weights/best.pt"
yolo = YOLO(text_det_model_path)

chars = 'abcdefghijklmnopqrstuvwxyz0123456789-'
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}


hidden_size = 256
n_layers = 3
dropout = 0.2
unfreeze_layers = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/content/drive/MyDrive/D:/ocr_model/ocr_crnn_model_best.pt"

crnn_model = CRNN(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout=dropout,
    unfreeze_layers=unfreeze_layers
).to(device)

crnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def ctc_decode(preds, idx_to_char, blank=0):
    """
    Decode logits từ mô hình CRNN sử dụng Greedy CTC decoding.

    Args:
        preds (Tensor): [B, T, V] logits đầu ra (chưa softmax hoặc log_softmax).
        idx_to_char (dict): Từ điển ánh xạ từ index về ký tự.
        blank (int): Index tương ứng với ký tự 'blank' trong CTC.

    Returns:
        List[str]: Danh sách các chuỗi đã decode.
    """
    pred_indices = preds.argmax(dim=2).cpu().numpy()  # [B, T]
    results = []

    for indices in pred_indices:
        prev = blank
        decoded = []
        for i in indices:
            if i != prev and i != blank:
                decoded.append(idx_to_char[i])
            prev = i
        results.append(''.join(decoded))

    return results

def text_detection(img_path, text_del_model):
  text_det_result = text_del_model.predict(img_path, verbose=False)

  result = text_det_result[0]
  bboxes = result.boxes.xyxy.tolist()
  classes = result.boxes.cls.tolist()
  confs = result.boxes.conf.tolist()
  names = result.names

  return bboxes, classes, names, confs

def text_recognition(img, data_transform, text_reg_model, idx_to_char, device):
  transformed_image = data_transform(img)
  transformed_image = transformed_image.unsqueeze(0).to(device)
  text_reg_model.eval()

  with torch.no_grad():
    logits = text_reg_model(transformed_image).detach().cpu()

  text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)

  return text

def visualize_detection(img, detections):
  plt.figure(figsize=(12, 8))
  plt.imshow(img)
  plt.axis("off")

  for bbox, detected_class, confidence, transcribed_text in detections:
    x1, x2, y1, y2 = bbox
    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor='red',
            linewidth=2
        ))
    plt.text(
          x1, y1 - 10,
          f"{detected_class}: {confidence:.2f}\n{transcribed_text}",
          color='red',
          fontsize=12
      )
    
    
def predict(img_path, data_transforms, text_del_model, text_reg_model, idx_to_char, device):
  bboxes, classes, names, confs = text_detection(img_path, text_del_model)

  img = Image.open(img_path)

  predictions = []

  for bbox, cls, conf in zip(bboxes, classes, confs):
    x1, y1, x2, y2 = bbox
    confidence = conf
    detected_class = cls
    name = names[int(cls)]

    cropped_img = img.crop((x1, y1, x2, y2))

    transcribed_text = text_recognition(
        cropped_img,
        data_transforms,
        text_reg_model,
        idx_to_char,
        device)

    predictions.append((bbox, name, confidence, transcribed_text))

  visualize_detection(img, predictions)
  return predictions


data_transform = {
    'train': transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(degrees=1, shear=1),
        transforms.RandomPerspective(
            distortion_scale=0.3,
            p=0.5,
            interpolation=3
        ),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]),
}


img_dir = "/dataset/apanar_06.08.2002/IMG_1247.JPG"
inf_transform = data_transform['val']

predict(img_dir,
        inf_transform,
        yolo,
        crnn_model,
        idx_to_char,
        device)