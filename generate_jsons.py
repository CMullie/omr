import re
import json
import pandas as pd
import os

intput_dir = "raw_data/ds2_complete/big_jsons/" #<- TO MODIFY
output_dir = "raw_data/ds2_complete/small_jsons/"#<- TO MODIFY

#FUNCTIONS
df_class_mapping = pd.read_csv("raw_data/class_mapping.csv")

def preprocess_class_lookup(df_class_mapping):
    return dict(zip(df_class_mapping['original_id'], df_class_mapping['class_rename']))

def get_note_count(annotations):
    lst_notes = []

    for elem in annotations:
        if int(elem["cat_id"][0]) >= 25 and int(elem["cat_id"][0]) <= 40:
            lst_notes.append(elem)

    return len(lst_notes)

def get_distinct_keys(annotations, class_dico):
    lst = []
    for elem in annotations:
        key = int(elem["cat_id"][0])
        key_string = get_class_name(key, class_dico)

        if key >= 6 and key <= 9 and key_string not in lst:
            lst.append(key_string)

    return lst

def get_object_distinct_count(annotations):
    lst = []
    for elem in annotations:
        if elem["cat_id"][0] not in lst:
            lst.append(elem["cat_id"][0])

    return len(lst)

def get_staff_count(annotations):
    count = 0
    for elem in annotations:
        if int(elem["cat_id"][0]) == 135:
            count = count + 1

    return count

def get_highest_note(annotations):
    target = 0
    for elem in annotations:
        if int(elem["cat_id"][0]) >= 25 and int(elem["cat_id"][0]) <= 40:
            target = max(int(elem["rel_position"]), target)

    return target

def get_lowest_note(annotations):
    target = 0
    for elem in annotations:
        if int(elem["cat_id"][0]) >= 25 and int(elem["cat_id"][0]) <= 40:
            target = min(int(elem["rel_position"]), target)

    return target

def get_time_signature(annotations, class_dico):
    lst = []
    for elem in annotations:
        key = int(elem["cat_id"][0])
        key_string = get_class_name(key, class_dico)

        if key >= 13 and key <= 24:
            lst.append(key_string)

    return lst[:2]

def get_class_name(id, class_dico):

    return class_dico.get(id, None)

def a_bbox_to_o_bbox(a_bbox):
    cord1 = a_bbox[0]
    cord2 = a_bbox[1]
    cord3 = a_bbox[2]
    cord4 = a_bbox[3]

    return [cord3, cord4, cord3, cord2, cord1, cord2, cord1, cord4]

def export_to_excel(df,name):
    df.to_excel(name, index=True)

def calcul_new_dim(dimension:list,rel_position:int):
    #x0 correspond a x_bas
    x0,y0,x1,y1=dimension

    mid=(y0+y1)/2
    hauter_box=y1-y0

    mid_position_zero=mid+hauter_box*rel_position/2
    y0_redim=mid_position_zero-110
    y1_redim=mid_position_zero+110

    a_bbox_redim=[x0-10,y0_redim,x1+10,y1_redim]
    return a_bbox_redim

# def get_possible_classes():
#     duration=[1,2,4,8,16,32,62,128]
#     possible_classes=[]

#     for i in range(-11,12):
#         for j in duration:
#             possible_classes.append(f"d{j}p{i}")

#     return possible_classes

def main():

    class_dico = preprocess_class_lookup(df_class_mapping)

    pattern = r'aug-(.*?)-'
    df_info=pd.DataFrame()
    dico_cat_count = {}
    # possible_classes = get_possible_classes()

    # Loop through all files in the folder
    for filename in os.listdir(intput_dir):
        if filename.endswith(".json"):  # Check if it's a JSON file
            file_path = os.path.join(intput_dir, filename)
            print(file_path)

            # Read the JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    d = json.load(file)  # Load JSON content
                    print(f"Contents of {filename}:")
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
            file.close()

            images = d['images']
            annotations = d['annotations']

            for i in range(len(images)):
                img = images[i]
                lst_ann = img["ann_ids"]
                lst_anno_rework = []

                for elem in lst_ann:
                    anno = annotations[elem]

                    duration = 0
                    pos = 0
                    is_note = anno["cat_id"][0] in("25", "27", "29", "31", "33", "35", "37", "39")

                    if is_note:
                        duration = anno["comments"].split(";")[1].replace("duration:", "")
                        duration = duration.replace(".", "")
                        pos = anno["comments"].split(";")[2].replace("rel_position:", "")
                        note_class = "d" + str(duration) + "p" + str(pos)
                        #index_class = dict_new_classes.get(note_class, None)

                    # if is_note and note_class not in possible_classes: #outlier we exclude
                    #     pass
                    # elif anno["cat_id"][0] not in("42", "122"):
                    dict_anno = {}

                    dict_anno["id"] = elem
                    #dict_anno["a_bbox_old"] = anno["a_bbox"]
                    dict_anno["a_bbox"] = anno["a_bbox"]
                    dict_anno["cat_id"] = anno["cat_id"].copy()
                    #dict_anno["cat_id_old"] = anno["cat_id"].copy()
                    dict_anno["area"] = anno["area"]

                    dict_anno["img_id"] = img["id"]
                    dict_anno["instance"] = anno["comments"].split(";")[0].replace("instance:", "")

                    if is_note:
                        dict_anno["duration"] = duration
                        dict_anno["rel_position"] = pos
                        #dict_anno["a_bbox"]=calcul_new_dim(anno["a_bbox"],int(pos))
                        #dict_anno["cat_id"][0] = index_class

                        # if abs(pos) > 15:
                        #     print(img["filename"])

                    if dict_anno["cat_id"][0] not in dico_cat_count.keys():
                        dico_cat_count[dict_anno["cat_id"][0]] = 1
                    else:
                        dico_cat_count[dict_anno["cat_id"][0]] = dico_cat_count[dict_anno["cat_id"][0]] + 1

                    dict_anno["o_bbox"] = a_bbox_to_o_bbox(dict_anno["a_bbox"])
                    lst_anno_rework.append(dict_anno)

                json_img = {}
                json_img["id"] = img["id"]
                json_img["filename"] = img["filename"]
                json_img["width"] = img["width"]
                json_img["height"] = img["height"]

                # Regular expression pattern to capture the word between 'aug-' and '-page'
                pattern = r'aug-(.*?)-'
                match = re.search(pattern, img["filename"])
                if match:
                    police = match.group(1)
                else:
                    police = "unknown"

                json_img["police"] = police
                json_img["nb_object"] = len(lst_anno_rework)
                json_img["nb_object_distinct"] = get_object_distinct_count(lst_anno_rework)
                json_img["nb_notes"] = get_note_count(lst_anno_rework)
                json_img["highest_note"] = get_highest_note(lst_anno_rework)
                json_img["lowest_note"] = get_lowest_note(lst_anno_rework)
                json_img["keys"] = get_distinct_keys(lst_anno_rework, class_dico)
                json_img["list_staff"] = get_staff_count(lst_anno_rework)
                json_img["time_signature"] = get_time_signature(lst_anno_rework, class_dico)
                json_img["page_number"] = img["filename"][-5]
                json_img["annotations"] = lst_anno_rework

                if i % 100 == 0:
                    print(i, " / ", len(images))

                #EXCEL
                data_without_annot=json_img.copy()
                del data_without_annot['annotations']
                new_row = pd.DataFrame([data_without_annot])
                df_info = pd.concat([df_info, new_row], ignore_index=True)

                with open(output_dir + json_img["filename"].replace(".png", ".json"), "w") as json_file:
                    json.dump(json_img, json_file, indent=4)  # indent=4 for pretty formatting

    #Export to excel
    export_to_excel(df_info, output_dir + "/image_info.xlsx")

if __name__ == "__main__":
    main()
