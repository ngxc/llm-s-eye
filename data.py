import json
import csv


coco_json_path = "captions_train2017.json"

output_csv_path = "coco_image_captions.csv"


with open(coco_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


images_dict = {img['id']: img['file_name'] for img in data['images']}


with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption'].strip()
        file_name = images_dict[image_id]
        writer.writerow([file_name, caption])

print(f"COCO captions have been converted to {output_csv_path}")
