import re
import json
import pandas as pd

output_dir = "raw_data/new_boxes_train/json/"

with open('raw_data/ds2_dense/deepscores_train.json') as json_data:
    d = json.load(json_data)
    json_data.close()

dict_new_classes = {'d1p-11': '209',
 'd2p-11': '210',
 'd4p-11': '211',
 'd8p-11': '212',
 'd16p-11': '213',
 'd32p-11': '214',
 'd62p-11': '215',
 'd128p-11': '216',
 'd1p-10': '217',
 'd2p-10': '218',
 'd4p-10': '219',
 'd8p-10': '220',
 'd16p-10': '221',
 'd32p-10': '222',
 'd62p-10': '223',
 'd128p-10': '224',
 'd1p-9': '225',
 'd2p-9': '226',
 'd4p-9': '227',
 'd8p-9': '228',
 'd16p-9': '229',
 'd32p-9': '230',
 'd62p-9': '231',
 'd128p-9': '232',
 'd1p-8': '233',
 'd2p-8': '234',
 'd4p-8': '235',
 'd8p-8': '236',
 'd16p-8': '237',
 'd32p-8': '238',
 'd62p-8': '239',
 'd128p-8': '240',
 'd1p-7': '241',
 'd2p-7': '242',
 'd4p-7': '243',
 'd8p-7': '244',
 'd16p-7': '245',
 'd32p-7': '246',
 'd62p-7': '247',
 'd128p-7': '248',
 'd1p-6': '249',
 'd2p-6': '250',
 'd4p-6': '251',
 'd8p-6': '252',
 'd16p-6': '253',
 'd32p-6': '254',
 'd62p-6': '255',
 'd128p-6': '256',
 'd1p-5': '257',
 'd2p-5': '258',
 'd4p-5': '259',
 'd8p-5': '260',
 'd16p-5': '261',
 'd32p-5': '262',
 'd62p-5': '263',
 'd128p-5': '264',
 'd1p-4': '265',
 'd2p-4': '266',
 'd4p-4': '267',
 'd8p-4': '268',
 'd16p-4': '269',
 'd32p-4': '270',
 'd62p-4': '271',
 'd128p-4': '272',
 'd1p-3': '273',
 'd2p-3': '274',
 'd4p-3': '275',
 'd8p-3': '276',
 'd16p-3': '277',
 'd32p-3': '278',
 'd62p-3': '279',
 'd128p-3': '280',
 'd1p-2': '281',
 'd2p-2': '282',
 'd4p-2': '283',
 'd8p-2': '284',
 'd16p-2': '285',
 'd32p-2': '286',
 'd62p-2': '287',
 'd128p-2': '288',
 'd1p-1': '289',
 'd2p-1': '290',
 'd4p-1': '291',
 'd8p-1': '292',
 'd16p-1': '293',
 'd32p-1': '294',
 'd62p-1': '295',
 'd128p-1': '296',
 'd1p0': '297',
 'd2p0': '298',
 'd4p0': '299',
 'd8p0': '300',
 'd16p0': '301',
 'd32p0': '302',
 'd62p0': '303',
 'd128p0': '304',
 'd1p1': '305',
 'd2p1': '306',
 'd4p1': '307',
 'd8p1': '308',
 'd16p1': '309',
 'd32p1': '310',
 'd62p1': '311',
 'd128p1': '312',
 'd1p2': '313',
 'd2p2': '314',
 'd4p2': '315',
 'd8p2': '316',
 'd16p2': '317',
 'd32p2': '318',
 'd62p2': '319',
 'd128p2': '320',
 'd1p3': '321',
 'd2p3': '322',
 'd4p3': '323',
 'd8p3': '324',
 'd16p3': '325',
 'd32p3': '326',
 'd62p3': '327',
 'd128p3': '328',
 'd1p4': '329',
 'd2p4': '330',
 'd4p4': '331',
 'd8p4': '332',
 'd16p4': '333',
 'd32p4': '334',
 'd62p4': '335',
 'd128p4': '336',
 'd1p5': '337',
 'd2p5': '338',
 'd4p5': '339',
 'd8p5': '340',
 'd16p5': '341',
 'd32p5': '342',
 'd62p5': '343',
 'd128p5': '344',
 'd1p6': '345',
 'd2p6': '346',
 'd4p6': '347',
 'd8p6': '348',
 'd16p6': '349',
 'd32p6': '350',
 'd62p6': '351',
 'd128p6': '352',
 'd1p7': '353',
 'd2p7': '354',
 'd4p7': '355',
 'd8p7': '356',
 'd16p7': '357',
 'd32p7': '358',
 'd62p7': '359',
 'd128p7': '360',
 'd1p8': '361',
 'd2p8': '362',
 'd4p8': '363',
 'd8p8': '364',
 'd16p8': '365',
 'd32p8': '366',
 'd62p8': '367',
 'd128p8': '368',
 'd1p9': '369',
 'd2p9': '370',
 'd4p9': '371',
 'd8p9': '372',
 'd16p9': '373',
 'd32p9': '374',
 'd62p9': '375',
 'd128p9': '376',
 'd1p10': '377',
 'd2p10': '378',
 'd4p10': '379',
 'd8p10': '380',
 'd16p10': '381',
 'd32p10': '382',
 'd62p10': '383',
 'd128p10': '384',
 'd1p11': '385',
 'd2p11': '386',
 'd4p11': '387',
 'd8p11': '388',
 'd16p11': '389',
 'd32p11': '390',
 'd62p11': '391',
 'd128p11': '392'}

#FUNCTIONS
df_class_mapping = pd.read_csv("raw_data/class_mapping.csv")

def preprocess_class_lookup(df_class_mapping):
    return dict(zip(df_class_mapping['original_id'], df_class_mapping['class_rename']))

def get_note_count(annotations):
    lst_notes = []

    for elem in annotations:
        if int(elem["cat_id_old"][0]) >= 25 and int(elem["cat_id_old"][0]) <= 40:
            lst_notes.append(elem)

    return len(lst_notes)

def get_distinct_keys(annotations, class_dico):
    lst = []
    for elem in annotations:
        key = int(elem["cat_id_old"][0])
        key_string = get_class_name(key, class_dico)

        if key >= 6 and key <= 9 and key_string not in lst:
            lst.append(key_string)

    return lst

def get_object_distinct_count(annotations):
    lst = []
    for elem in annotations:
        if elem["cat_id_old"][0] not in lst:
            lst.append(elem["cat_id_old"][0])

    return len(lst)

def get_staff_count(annotations):
    count = 0
    for elem in annotations:
        if int(elem["cat_id_old"][0]) == 135:
            count = count + 1

    return count

def get_highest_note(annotations):
    target = 0
    for elem in annotations:
        if int(elem["cat_id_old"][0]) >= 25 and int(elem["cat_id_old"][0]) <= 40:
            target = max(int(elem["rel_position"]), target)

    return target

def get_lowest_note(annotations):
    target = 0
    for elem in annotations:
        if int(elem["cat_id_old"][0]) >= 25 and int(elem["cat_id_old"][0]) <= 40:
            target = min(int(elem["rel_position"]), target)

    return target

def get_time_signature(annotations, class_dico):
    lst = []
    for elem in annotations:
        key = int(elem["cat_id_old"][0])
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

def get_possible_classes():
    duration=[1,2,4,8,16,32,62,128]
    possible_classes=[]

    for i in range(-11,12):
        for j in duration:
            possible_classes.append(f"d{j}p{i}")

    return possible_classes

def main():

    images = d['images']
    annotations = d['annotations']
    class_dico = preprocess_class_lookup(df_class_mapping)

    pattern = r'aug-(.*?)-'
    df_info=pd.DataFrame()
    dico_cat_count = {}
    possible_classes = get_possible_classes()

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
                index_class = dict_new_classes.get(note_class, None)

            if is_note and note_class not in possible_classes: #outlier we exclude
                pass
            elif anno["cat_id"][0] not in("42", "122"):
                dict_anno = {}

                dict_anno["id"] = elem
                dict_anno["a_bbox_old"] = anno["a_bbox"]
                dict_anno["a_bbox"] = anno["a_bbox"]
                dict_anno["cat_id"] = anno["cat_id"].copy()
                dict_anno["cat_id_old"] = anno["cat_id"].copy()
                dict_anno["area"] = anno["area"]

                dict_anno["img_id"] = img["id"]
                dict_anno["instance"] = anno["comments"].split(";")[0].replace("instance:", "")

                if is_note:
                    dict_anno["duration"] = duration
                    dict_anno["rel_position"] = pos
                    dict_anno["a_bbox"]=calcul_new_dim(anno["a_bbox"],int(pos))
                    dict_anno["cat_id"][0] = index_class

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
