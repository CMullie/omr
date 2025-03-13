import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from shapely.geometry import Polygon, LineString
import warnings

# Convertir les catégories en DataFrame
def convert_categories_to_df(categories_json):
    #categories_data = big_json['categories']
    categories_df = pd.DataFrame.from_dict(categories_json, orient='index')
    return categories_df

# Convertir les images en DataFrame
def convert_images_to_df(partition_json):
    filtered_data = {key: value for key, value in partition_json.items() if key != 'annotations'}
    print(filtered_data)
    #image_df = pd.DataFrame.from_dict(filtered_data, orient='columns')
    image_df = pd.DataFrame([filtered_data])
    print(image_df)
    #image_df.set_index("id", inplace=True)
    return image_df.head(1) #je comprends pas pq mais j'ai deux lignes identiques donc je garde la première

def convert_annotations_to_df(parititon_json_annotations):
    anns_df = pd.DataFrame(parititon_json_annotations)
    #anns_df.set_index("id", inplace=True)
    return anns_df

# Créer une table de mappage des classes pour YOLO
def create_class_mapping(categories_df):
    # Filtrer pour ne garder que les catégories DeepScores (pas les doublons de muscima++)
    deepscores_categories = categories_df[categories_df['annotation_set'] == 'deepscores']

    # Créer un dictionnaire de mappage des ID originaux vers des ID séquentiels pour YOLO
    # YOLO utilise des ID de classe qui commencent à 0 et sont séquentiels
    class_map = {}
    class_names = []

    # Trier par ID numérique
    sorted_cats = deepscores_categories.sort_index(key=lambda x: x.astype(int))

    # Créer un nouveau mapping séquentiel
    for i, (cat_id, row) in enumerate(sorted_cats.iterrows()):
        class_map[cat_id] = i
        class_names.append(row['name'])

    # Ajouter aussi les catégories muscima++ qui n'ont pas d'équivalent dans deepscores
    muscima_categories = categories_df[categories_df['annotation_set'] == 'muscima++']
    for cat_id, row in muscima_categories.iterrows():
        # Vérifier si le nom de cette catégorie existe déjà dans notre liste
        name = row['name']
        matching_deepscores = deepscores_categories[deepscores_categories['name'].str.lower() == name.lower()]

        if matching_deepscores.empty:
            # Cette catégorie muscima++ n'a pas d'équivalent, l'ajouter
            class_map[cat_id] = len(class_names)
            class_names.append(name)

    return class_map, class_names

# Sauvegarder le mappage des classes pour référence future
def save_class_mapping(class_map, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Créer un DataFrame pour le mappage
    mapping_df = pd.DataFrame({
        'original_id': list(class_map.keys()),
        'yolo_id': list(class_map.values()),
        'class_name': [class_names[yolo_id] for yolo_id in class_map.values()]
    })

    # Trier par ID YOLO
    mapping_df = mapping_df.sort_values('yolo_id')

    # Sauvegarder au format CSV
    mapping_df.to_csv(os.path.join(output_dir, 'class_mapping.csv'), index=False)

    # Sauvegarder les noms de classes au format attendu par YOLO
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"Mappage des classes sauvegardé dans {output_dir}/class_mapping.csv")
    print(f"Noms des classes sauvegardés dans {output_dir}/classes.txt")

# Fonction améliorée pour convertir une boîte englobante orientée en format YOLO
def obb_to_yolo(obb, img_width, img_height):
    # Vérifier si les coordonnées forment un polygone fermé
    if len(obb) != 8:
        return None

    # Gérer le cas des boîtes dégénérées où certains points sont identiques
    points = [(obb[i], obb[i + 1]) for i in range(0, len(obb), 2)]

    # Filtrer les points uniques
    unique_points = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)

    # Si moins de 3 points uniques, la boîte est invalide
    if len(unique_points) < 3:
        return None

    # Si nous avons exactement 3 points uniques, ajouter un 4ème point pour former un quadrilatère
    if len(unique_points) == 3:
        # Créer un point symétrique par rapport au centre des 3 points
        center_x = sum(p[0] for p in unique_points) / 3
        center_y = sum(p[1] for p in unique_points) / 3

        # Le 4ème point est symétrique au point opposé du centre
        p0 = unique_points[0]
        fourth_point = (2 * center_x - p0[0], 2 * center_y - p0[1])
        unique_points.append(fourth_point)

    try:
        # Essayer de créer un polygone avec les points uniques
        polygon = Polygon(unique_points)

        # Si le polygone n'est pas valide, essayer de le rendre valide
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

            # Si toujours pas valide, abandonner
            if not polygon.is_valid:
                return None

        # Obtenir le rectangle minimum orienté
        min_rect = polygon.minimum_rotated_rectangle

        # Si nous n'avons pas de rectangle, abandonner
        if min_rect is None or min_rect.is_empty:
            return None

        # Extraire les coins
        corners = np.array(min_rect.exterior.coords[:-1])  # Exclure le dernier point (identique au premier)

        # S'assurer que nous avons 4 coins
        if len(corners) != 4:
            return None

        # Calculer la longueur des côtés
        edges = [np.linalg.norm(corners[i] - corners[(i+1)%4]) for i in range(4)]

        # Trouver les deux côtés adjacents
        width = max(edges)
        height = min(edges)

        # Obtenir le centre du rectangle
        center = min_rect.centroid.coords[0]
        center_x, center_y = center

        # Calculer l'angle du rectangle
        # Nous prenons le côté le plus long comme référence pour l'angle
        longest_edge_idx = edges.index(width)
        angle = np.rad2deg(np.arctan2(
            corners[(longest_edge_idx+1)%4][1] - corners[longest_edge_idx][1],
            corners[(longest_edge_idx+1)%4][0] - corners[longest_edge_idx][0]
        ))

        # Normaliser les coordonnées pour le format YOLO
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        # Retourner aussi la hauteur non normalisée pour l'utiliser comme hauteur de note
        original_height = min(edges)

        return [center_x, center_y, width, height, angle, original_height]

    except Exception as e:
        print(f"Erreur lors de la conversion OBB->YOLO: {e}, OBB: {obb}")
        return None

# Sauvegarder les annotations YOLO dans un fichier texte avec les classes mappées et hauteur de note
def save_yolo_annotations(anns_df, images_df, categories_df, output_label_dir, class_map, class_names):
    os.makedirs(output_label_dir, exist_ok=True)

    # Créer un dictionnaire pour regrouper les annotations par image
    image_annotations = {}
    stats = {"total": 0, "success": 0, "fail": 0}

    # Pour chaque annotation
    for idx, ann in anns_df.iterrows():
        img_id = ann['img_id']

        # Si l'ID de l'image n'est pas encore dans le dictionnaire, l'initialiser
        if img_id not in image_annotations:
            image_annotations[img_id] = []

        # Obtenir l'ID de catégorie original
        orig_category_id = str(ann['cat_id'][0])
        stats["total"] += 1

        # Vérifier si cette catégorie est dans notre mappage
        if orig_category_id not in class_map:
            print(f"⚠️ Catégorie {orig_category_id} non trouvée dans le mappage des classes.")
            stats["fail"] += 1
            continue

        # Obtenir l'ID de classe YOLO mappé
        yolo_class_id = class_map[orig_category_id]

        # Obtenir le nom de la classe
        class_name = class_names[yolo_class_id]

        obb = ann['o_bbox']

        # Trouver les dimensions de l'image
        img_row = images_df[images_df['id'] == img_id]
        if img_row.empty:
            print(f"⚠️ Image ID {img_id} non trouvée dans le DataFrame des images.")
            stats["fail"] += 1
            continue

        img_width = img_row['width'].iloc[0]
        img_height = img_row['height'].iloc[0]

        # Convertir la boîte englobante en format YOLO (ajout de la hauteur originale)
        yolo_bbox = obb_to_yolo(obb, img_width, img_height)

        # Si la conversion a réussi, ajouter à la liste des annotations pour cette image
        if yolo_bbox is not None:
            # Séparons la hauteur originale du reste des coordonnées YOLO
            note_height = yolo_bbox[5]  # La hauteur originale en pixels
            yolo_coords = yolo_bbox[:5]  # Les coordonnées YOLO standard

            # Ajouter les informations à notre liste d'annotations
            image_annotations[img_id].append((yolo_class_id, yolo_coords, class_name, note_height))
            stats["success"] += 1
        else:
            stats["fail"] += 1

    # Pour chaque image, écrire toutes ses annotations dans un fichier
    file_count = 0
    for img_id, annotations in image_annotations.items():
        if not annotations:
            continue

        img_row = images_df[images_df['id'] == img_id]
        img_name = img_row['filename'].iloc[0]
        label_path = os.path.join(output_label_dir, img_name.replace(".png", ".txt"))

        with open(label_path, "w") as f:  # 'w' au lieu de 'a' pour écraser le fichier existant
            for yolo_class_id, yolo_coords, class_name, note_height in annotations:
                # Format: class_id x y width height angle class_name note_height
                #f.write(f"{yolo_class_id} {' '.join(map(str, yolo_coords))} # {class_name} {note_height:.2f}\n")
                f.write(f"{yolo_class_id} {' '.join(map(str, yolo_coords))}\n")

        file_count += 1

    print(f"Statistiques de conversion:")
    print(f"  - Annotations totales: {stats['total']}")
    print(f"  - Conversions réussies: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"  - Conversions échouées: {stats['fail']} ({stats['fail']/stats['total']*100:.1f}%)")
    print(f"  - Fichiers d'annotation générés: {file_count}")

# Convertir les coordonnées YOLO en coordonnées de coins
def yolo_to_corners(center_x, center_y, width, height, rotation, img_width, img_height):
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    rotation = np.deg2rad(rotation)
    corners = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])
    R = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])
    rot_corners = np.dot(corners, R)
    rot_corners[:, 0] += center_x
    rot_corners[:, 1] += center_y
    return rot_corners

def draw_yolo_boxes(anns_df, images_df, categories_df, output_image_dir, input_image_dir, class_map, class_names):
    os.makedirs(output_image_dir, exist_ok=True)

    # Créer un dictionnaire pour regrouper les annotations par image
    image_annotations = {}

    # Créer une palette de couleurs pour différencier les classes
    # On utilise une fonction de hachage pour avoir des couleurs distinctes mais consistantes
    def get_color(class_id):
        # On génère une couleur à partir de l'ID de classe
        np.random.seed(class_id * 10)
        return np.random.rand(3)

    # Pour chaque annotation
    for idx, ann in anns_df.iterrows():
        img_id = ann['img_id']

        # Si l'ID de l'image n'est pas encore dans le dictionnaire, l'initialiser
        if img_id not in image_annotations:
            image_annotations[img_id] = []

        # Obtenir l'ID de catégorie original
        orig_category_id = str(ann['cat_id'][0])

        # Vérifier si cette catégorie est dans notre mappage
        if orig_category_id not in class_map:
            continue

        # Obtenir l'ID de classe YOLO mappé
        yolo_class_id = class_map[orig_category_id]

        # Obtenir le nom de la classe
        class_name = class_names[yolo_class_id]

        obb = ann['o_bbox']

        # Trouver les dimensions de l'image
        img_row = images_df[images_df['id'] == img_id]
        if img_row.empty:
            continue

        img_width = img_row['width'].iloc[0]
        img_height = img_row['height'].iloc[0]

        # Convertir la boîte englobante en format YOLO
        yolo_bbox = obb_to_yolo(obb, img_width, img_height)

        # Si la conversion a réussi, ajouter à la liste des annotations pour cette image
        if yolo_bbox is not None:
            # Séparons la hauteur originale du reste des coordonnées YOLO
            note_height = yolo_bbox[5]  # La hauteur originale en pixels
            yolo_coords = yolo_bbox[:5]  # Les coordonnées YOLO standard

            image_annotations[img_id].append((yolo_class_id, yolo_coords, class_name, note_height))

    # Pour chaque image, dessiner toutes ses annotations
    for img_id, annotations in image_annotations.items():
        if not annotations:
            continue

        img_row = images_df[images_df['id'] == img_id]
        img_name = img_row['filename'].iloc[0]
        img_width = img_row['width'].iloc[0]
        img_height = img_row['height'].iloc[0]

        # Le chemin complet de l'image d'entrée
        img_path = os.path.join(input_image_dir, img_name)

        # Vérifier si l'image existe
        if not os.path.exists(img_path):
            print(f"⚠️ Image {img_path} non trouvée.")
            continue

        try:
            # Ouvrir l'image
            img = Image.open(img_path)

            # Créer une figure matplotlib
            fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
            ax.imshow(img)

            # Dessiner chaque boîte englobante
            for class_id, yolo_coords, class_name, note_height in annotations:
                # Obtenir la couleur pour cette classe
                color = get_color(class_id)

                # Convertir les coordonnées YOLO en coins
                corners = yolo_to_corners(yolo_coords[0], yolo_coords[1], yolo_coords[2], yolo_coords[3], yolo_coords[4], img_width, img_height)

                # Dessiner le polygone
                polygon = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1)
                ax.add_patch(polygon)

                # Ajouter une étiquette avec le nom de la classe et la hauteur de note
                # Utiliser le premier coin comme position pour l'étiquette
                # label_x, label_y = corners[0]
                # label_text = f"{class_name} (h:{note_height:.1f}px)"
                # ax.text(label_x, label_y, label_text, fontsize=8, color=color,
                #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

            # Configurer les axes
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)
            ax.axis('off')  # Cacher les axes

            # Sauvegarder l'image annotée
            output_path = os.path.join(output_image_dir, img_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)

            print(f"Image annotée sauvegardée: {output_path}")

        except Exception as e:
            print(f"Erreur lors de l'annotation de {img_name}: {e}")

def main():

    #SETUP
    with open('raw_data/new_classes.json', 'r') as file:
        categories_json = json.load(file)
    file.close()

    output_dir = "raw_data/new_boxes_complete" #<- TO MODIFY
    json_folder = os.path.join(output_dir, "json")
    output_label_dir = os.path.join(output_dir, "labels")
    output_image_dir = os.path.join(output_dir, "images")
    input_image_dir = "raw_data/ds2_dense/images"  # <-TO MODIFY

    #CATEGORIES
    categories_df = convert_categories_to_df(categories_json)
    class_map, class_names = create_class_mapping(categories_df)
    print(class_map)
    save_class_mapping(class_map, class_names, output_dir)

    #LOOP HERE
    i = 0
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):  # Ensure it's a JSON file
            i += 1
            if i % 1 == 0:
                print("----------------------------------------")
                print(i, " / ")
                print("----------------------------------------")

            partition_json_file = os.path.join(json_folder, filename)

            #partition_json_file = "raw_data/Data_JSON/lg-3948783-aug-beethoven--page-1.json"

            with open(partition_json_file, "r") as file:
                partition_json = json.load(file)
            file.close()

            #print(partition_json)

            image_df = convert_images_to_df(partition_json)
            anns_df = convert_annotations_to_df(partition_json["annotations"])

            print("\nSauvegarde des annotations YOLO avec les classes et hauteurs de notes...")
            save_yolo_annotations(anns_df, image_df, categories_df, output_label_dir, class_map, class_names)

            draw_yolo_boxes(anns_df, image_df, categories_df, output_image_dir, input_image_dir, class_map, class_names)


if __name__ == "__main__":
    main()
