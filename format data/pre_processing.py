import os
import shutil
import xml.etree.ElementTree as ET
import yaml


def extract_data_from_xml(words_xml_path):

    tree = ET.parse(words_xml_path)
    root = tree.getroot()

    output_path = []
    for image in root:
        outputs = []
        for bbs in image.findall('taggedRectangles'):
            for bb in bbs:
                output = {}
                bbox = [
                    float(bb.attrib['x']),
                    float(bb.attrib['y']),
                    float(bb.attrib['width']),
                    float(bb.attrib['height'])
                ]
                output["bbox"] = bbox
                output["label"]= bb[0].text.lower()
                output["size"] = [float(image[1].attrib['x']),
                                float(image[1].attrib['y'])]

                outputs.append(output)
        output_path.append((image[0].text, outputs))

    return output_path



def convert_to_yolov8_format(outputs):

    yolov8_data = []

    for op in outputs:
        yolov8_labels = []
        for item in op[1]:
            x, y, w, h = item["bbox"]

            # Normalize
            x_center = (x + w / 2) / item["size"][0]
            y_center = (y + h / 2) / item["size"][1]
            nor_w = w / item["size"][0]
            nor_h = h / item["size"][1]

            class_id = 0

            yolov8_label = f"{class_id} {x_center:.6f} {y_center:.6f} {nor_w:.6f} {nor_h:.6f}"
            yolov8_labels.append(yolov8_label)
        yolov8_data.append((op[0], yolov8_labels))
    return yolov8_data

def save_data(data, src_img_dir, save_dir):
    # create folder if not exists
    os.makedirs(save_dir, exist_ok=True)

    # make images and labels folder
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for image_path, yolov8_label in data:
        # copy image to image folder
        shutil.copy(
            os.path.join(src_img_dir, image_path),
            os.path.join(save_dir, "images")
        )

        # save label to label folder

        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        with open(os.path.join(save_dir, 'labels', f"{image_name}.txt"), 'w') as f:
            for label in yolov8_label:
                f.write(f"{label}\n")
                
if __name__ == "__main__":

    # Define paths
    words_xml_path = "data/words.xml"
    src_img_dir = "data/images"
    save_dir = "data/yolov8"

    # Extract data from XML
    outputs = extract_data_from_xml(words_xml_path)

    # Convert to YOLOv8 format
    yolov8_data = convert_to_yolov8_format(outputs)

    class_labels = ["text"]
    save_data_yaml_path = "/content/drive/MyDrive/D:/yolo_data"

    data_yaml = {
        'path': '/content/drive/MyDrive/D:/yolo_data',
        'test': 'test/images',
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': class_labels
    }

    yolo_yaml_path = os.path.join(
        save_data_yaml_path,
        'data.yml'
    )

    with open(yolo_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)