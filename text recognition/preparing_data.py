def split_bouding_boxes(output, save_dir, dataset_dir):
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    labels = []

    for image_path, annotations in output:
        full_image_path = os.path.join(dataset_dir, image_path)
        img = Image.open(full_image_path)
        for target in annotations:
            cropped_img = img.crop((target['bbox'][0],
                                    target['bbox'][1],
                                    target['bbox'][0]+target['bbox'][2],
                                    target['bbox'][1]+target['bbox'][3]))
            # filterout if 90% of the cropimage is back or whie
            if np.mean(cropped_img) < 35 or np.mean(cropped_img)>220:
                continue
            # check width and height of image
            if cropped_img.size[0] < 35 or cropped_img.size[1]>220:
                continue
            if len(target['label']) < 3:
                continue
            file_name = f"{count:06d}.jpg"
            cropped_img.save(os.path.join(save_dir, file_name))

            new_img_path = os.path.join(save_dir, file_name)
            label_text = f"{new_img_path}\t{target['label']}"
            labels.append(label_text)

            count += 1

    with open(os.path.join(save_dir, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
            
if __name__ == "__main__":
    dataset_dir = "/dataset/dataset"
    save_dir = "/dataset/split_dataset"
    src_img_dir = os.path.join(dataset_dir, "images")
    # xml_path = os.path.join(dataset_dir, "annotations", "train.xml")
    # output_path = parse_xml(xml_path)
    # yolov8_data = convert_to_yolov8_format(output_path)
    # save_data(yolov8_data, src_img_dir, save_dir)
    split_bouding_boxes(output_path, save_dir, dataset_dir)
    
    root_dir = save_dir

    img_paths = []
    labels = []

    with open(os.path.join(root_dir, 'labels.txt'), 'r') as f:
        for label in f:
            labels.append(label.strip().split('\t')[1])
            img_paths.append(label.strip().split('\t')[0])
    print(f"Total images: {len(img_paths)}")

    
    letter = []
    for char in labels:
        char.lower()
        letter.append(char)

    letter = "".join(letter)
    letter = sorted((set(letter)))

    chars = "".join(letter)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
    vocab_size = len(chars)
    print(f'Vocab: {chars}')
    print(f'Vocab size: {vocab_size}')