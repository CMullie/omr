import json
import os
from music_structure import Note, Measure, Staff, Score, TimeSignature, KeySignature
from midi_converter import convert_score_to_midi

def get_list_of_classes(detections, wanted_classes):
    list_output = []

    if type(wanted_classes) == str:
        wanted_classes = [wanted_classes]

    for elem in detections:
        if elem["class_name"] in wanted_classes:
            list_output.append(elem)

    return list_output

def attribute_staff_to_object(list_staff, object):
    closest_staff = 0
    shortest_center_distance = 10_000
    staff_index = -1

    for staff in list_staff:
        staff_index += 1
        staff_object_distance = abs(staff["y_center"] - object["y_center"])
        if staff_object_distance < shortest_center_distance:
            closest_staff = staff_index
            shortest_center_distance = staff_object_distance

    object["attributed_staff"] = closest_staff
    return object

def attribute_staffs_to_list(list_staff, list_objects, sort_by_staff=True):
    for i in range(len(list_objects)):
        list_objects[i] = attribute_staff_to_object(list_staff, list_objects[i])

    if sort_by_staff:
        list_objects = sorted(list_objects, key=lambda x: (x['attributed_staff'], x['x_center']))

    return list_objects

def attribute_relative_position_to_a_note(note, list_staff):
    staff = list_staff[note["attributed_staff"]]
    average_staff_height = sum(obj["height"] for obj in list_staff) / len(list_staff)
    interval_height = average_staff_height / 8

    dict_distance_pos = {}

    for i in range(-20, 21):
        dict_distance_pos[i] = abs(staff["y_center"] + (-i * interval_height) - note["y_center"])

    closest_position = min(dict_distance_pos, key=dict_distance_pos.get)
    note["relative_position"] = closest_position

    return note

def attribute_relative_key_to_a_note(note, list_keys):
    list_keys_on_staff = []
    note_staff_index = note["attributed_staff"]

    for elem in list_keys:
        if elem["attributed_staff"] == note_staff_index:
            list_keys_on_staff.append(elem)

    attributed_key = "unknown"
    for elem in list_keys_on_staff:
        if elem["x_center"] < note["x_center"]:
            attributed_key = elem["class_name"]

    note["attributed_key"] = attributed_key
    return note

def attribute_duration_to_a_note(note, list_beams, list_flags):
    note_duration = -1

    if note["class_name"] in ["noteheadWholeOnLine", "noteheadWholeInSpace"]:
        note_duration = 1

    if note["class_name"] in ["noteheadHalfOnLine", "noteheadHalfInSpace"]:
        note_duration = 2

    margin = 1.50
    if note["class_name"] in ["noteheadBlackOnLine", "noteheadBlackInSpace"]:
        margin_x1 = note["x_center"] - (note["width"] * margin/2)
        margin_x2 = note["x_center"] + (note["width"] * margin/2)

        # FLAGS
        has_8th_flag = False
        for elem in list_flags:
            if elem["attributed_staff"] == note["attributed_staff"]:
                is_totally_left_of_x1 = elem["x2"] < margin_x1
                is_totally_right_of_x2 = elem["x1"] > margin_x2
                if is_totally_left_of_x1 or is_totally_right_of_x2:
                    pass
                else:
                    has_8th_flag = True

        if has_8th_flag:
            note_duration = 8
        else:
            # BEAMS
            count_beams = 0
            for elem in list_beams:
                if elem["attributed_staff"] == note["attributed_staff"]:
                    is_totally_left_of_x1 = elem["x2"] < margin_x1
                    is_totally_right_of_x2 = elem["x1"] > margin_x2
                    if is_totally_left_of_x1 or is_totally_right_of_x2:
                        pass
                    else:
                        count_beams += 1

            if count_beams > 0:
                note_duration = 2 ** (2 + count_beams)  # ONE BEAM = 1/8, TWO BEAMS = 1/16
            else:
                note_duration = 4  # JUST A BLACK

    note["attributed_duration"] = note_duration
    return note

def attribute_characteristics_to_notes(list_notes, list_staff, list_keys, list_beams, list_flags):
    for i in range(len(list_notes)):
        list_notes[i] = attribute_relative_position_to_a_note(list_notes[i], list_staff)
        list_notes[i] = attribute_relative_key_to_a_note(list_notes[i], list_keys)
        list_notes[i] = attribute_duration_to_a_note(list_notes[i], list_beams, list_flags)

    return list_notes

def attribute_order_index(list_musical_objects):
    list_musical_objects = sorted(list_musical_objects, key=lambda x: (x['attributed_staff'], x['x_center']))

    for i in range(len(list_musical_objects)):
        current_object = list_musical_objects[i]
        previous_object = list_musical_objects[i - 1] if i > 0 else None

        # FIRST OBJECT
        if i == 0:
            order_index = 0
        else:
            # NEXT STAFF, SO ORDER IS 0 FOR FIRST OBJECT
            if current_object["attributed_staff"] > previous_object["attributed_staff"]:
                order_index = 0
            else:
                is_a_rest = current_object["class_name"][:4] == "rest"
                is_after_rest = previous_object["class_name"][:4] == "rest"
                are_objects_far = previous_object["x_center"] + (min(previous_object["width"], current_object["width"]) * 1.5) < current_object["x_center"]
                # THE OBJECTS ARE FAR ENOUGH TO BE CONSIDERED NOT ON THE SAME TIME
                if is_a_rest or is_after_rest or are_objects_far:
                    order_index = previous_object["order_index"] + 1
                # THE OBJECTS ARE CLOSE ENOUGH TO BE CONSIDERED ON THE SAME TIME
                else:
                    order_index = previous_object["order_index"]

        list_musical_objects[i]["order_index"] = order_index

    return list_musical_objects

def clef_position_to_midi(relative_position, clef_type):
    if clef_type == 'clefG':
        position_map = {
            # Positions au-dessus de la ligne médiane
            10: 88,   # Mi6
            9: 86,    # Ré6
            8: 84,    # Do6
            7: 83,    # Si5
            6: 81,    # La5
            5: 79,    # Sol5
            4: 77,    # Fa5
            3: 76,    # Mi5
            2: 74,    # Ré5
            1: 72,    # Do5
            # Position médiane
            0: 71,    # Si4 (B4)
            # Positions en-dessous de la ligne médiane
            -1: 69,   # La4
            -2: 67,   # Sol4
            -3: 65,   # Fa4
            -4: 64,   # Mi4
            -5: 62,   # Ré4
            -6: 60,   # Do4 (C4)
            -7: 59,   # Si3
            -8: 57,   # La3
            -9: 55,   # Sol3
            -10: 53,  # Fa3
        }
    elif clef_type == 'clefF':
        position_map = {
            # Positions au-dessus de la ligne médiane
            10: 67,   # Sol4
            9: 65,    # Fa4
            8: 64,    # Mi4
            7: 62,    # Ré4
            6: 60,    # Do4
            5: 59,    # Si3
            4: 57,    # La3
            3: 55,    # Sol3
            2: 53,    # Fa3
            1: 52,    # Mi3
            # Position médiane
            0: 50,    # Ré3 (D3)
            # Positions en-dessous de la ligne médiane
            -1: 48,   # Do3
            -2: 47,   # Si2
            -3: 45,   # La2
            -4: 43,   # Sol2
            -5: 41,   # Fa2
            -6: 40,   # Mi2
            -7: 38,   # Ré2
            -8: 36,   # Do2
            -9: 35,   # Si1
            -10: 33,  # La1
        }
    else:
        return clef_position_to_midi(relative_position, 'clefG')

    if relative_position in position_map:
        return position_map[relative_position]
    else:
        closest_pos = min(position_map.keys(), key=lambda x: abs(x - relative_position))
        offset = relative_position - closest_pos
        estimated_pitch = position_map[closest_pos] - int(offset * 1.5)
        return max(0, min(127, estimated_pitch))

def count_key_signatures(list_keys_signatures):
    staff_key_counts = {}

    for key in list_keys_signatures:
        staff_index = key.get('attributed_staff', 0)
        if staff_index not in staff_key_counts:
            staff_key_counts[staff_index] = {'keyFlat': 0, 'keySharp': 0}

        key_type = key.get('class_name', '')
        if key_type in ['keyFlat', 'keySharp'] and key.get('confidence', 0) >= 0.5:
            staff_key_counts[staff_index][key_type] += 1

    return staff_key_counts

def create_midi(json_file, output_file):
    print(f"Lecture du fichier {json_file}")
    try:
        with open(json_file, 'r') as f:
            # Vérifier si c'est déjà un fichier traité
            data = json.load(f)
            if not all(key in data[0] for key in ['attributed_staff', 'relative_position', 'attributed_key', 'attributed_duration']):
                print("Fichier de détection brut détecté, traitement en cours...")
                detections = data
                list_staff = get_list_of_classes(detections, "staff")
                list_staff = sorted(list_staff, key=lambda x: x['y1'])

                notes_labels = ["noteheadBlackInSpace", "noteheadBlackOnLine", "noteheadHalfInSpace", "noteheadHalfOnLine"]
                list_notes = get_list_of_classes(detections, notes_labels)
                list_notes = [obj for obj in list_notes if obj['confidence'] >= 0.6]

                rests_labels = ["restWhole", "restQuarter", "rest8th"]
                list_rests = get_list_of_classes(detections, rests_labels)
                list_rests = [obj for obj in list_rests if obj['confidence'] >= 0.6]

                keys_labels = ["clefF", "clefG"]
                list_keys = get_list_of_classes(detections, keys_labels)

                keys_signatures_labels = ["keyFlat", "keySharp"]
                list_keys_signatures = get_list_of_classes(detections, keys_signatures_labels)
                list_keys_signatures = [obj for obj in list_keys_signatures if obj['confidence'] >= 0.5]

                list_beams = get_list_of_classes(detections, "beam")
                list_beams = [obj for obj in list_beams if obj['confidence'] >= 0.6]

                list_flags = get_list_of_classes(detections, ["flag8thUp", "flag8thDown"])

                # Attribuer les portées aux objets
                list_notes = attribute_staffs_to_list(list_staff, list_notes)
                list_rests = attribute_staffs_to_list(list_staff, list_rests)
                list_keys = attribute_staffs_to_list(list_staff, list_keys)
                list_keys_signatures = attribute_staffs_to_list(list_staff, list_keys_signatures)
                list_beams = attribute_staffs_to_list(list_staff, list_beams)
                list_flags = attribute_staffs_to_list(list_staff, list_flags)

                # Attribuer des caractéristiques aux notes
                list_notes = attribute_characteristics_to_notes(list_notes, list_staff, list_keys, list_beams, list_flags)

                # Fusionner les notes et les silences
                list_musical_objects = list_notes + list_rests

                # Attribuer l'ordre des objets
                data = attribute_order_index(list_musical_objects)

                # Compter les altérations par portée
                key_counts = count_key_signatures(list_keys_signatures)

                print(f"Altérations détectées {key_counts[0]}")
            else:
                # Si les données sont déjà traitées, on vérifie si on a des informations sur les armures
                key_counts = {}


                print(f"Traitement terminé: {len(data)} objets musicaux préparés.")
    except Exception as e:
        print(f"Erreur lors du chargement ou du traitement du fichier: {e}")
        raise

    # Étape 2: Créer le fichier MIDI

    # Déterminer l'armure à partir des altérations détectées sur la première portée (index 0)
    flats = 0
    sharps = 0

    if 0 in key_counts:
        flats = key_counts[0].get('keyFlat', 0)
        sharps = key_counts[0].get('keySharp', 0)

    if flats > 0 and sharps > 0:
        print(f"Attention: {flats} bémols et {sharps} dièses détectés. Utilisation des plus nombreux.")
        if flats >= sharps:
            sharps = 0
        else:
            flats = 0

    # Créer l'armure
    key_signature = None
    if sharps > 0:
        key_signature = KeySignature(sharps=sharps, start_time=0)
        key_info = f"{sharps} dièse(s)"
    elif flats > 0:
        key_signature = KeySignature(sharps=-flats, start_time=0)  # Les bémols sont représentés par des valeurs négatives
        key_info = f"{flats} bémol(s)"
    else:
        key_info = "aucune armure"
    print(f"Armure utilisée: {key_info}")

    # Filtrer les notes
    treble_notes = [n for n in data if 'notehead' in n.get('class_name', '') and n.get('attributed_key') == 'clefG']
    bass_notes = [n for n in data if 'notehead' in n.get('class_name', '') and n.get('attributed_key') == 'clefF']

    print(f"Trouvé {len(treble_notes)} notes en clef de sol et {len(bass_notes)} notes en clef de fa")

    # Créer la partition avec l'armure
    score = Score(tempo=79, key_signature=key_signature)  # tempo de 79 par défaut

    # Portée en clef de sol
    if len(treble_notes) > 0:
        treble_staff = Staff("treble")
        treble_measure = Measure(start_time=0)

        current_time = 0.0
        duration = 0
        pos = 0
        for i, note_data in enumerate(treble_notes):
            current_time += duration if note_data.get('order_index', 0) != pos else 0
            # Calculer la hauteur
            clef_type = note_data.get('attributed_key', 'clefG')
            relative_pos = note_data.get('relative_position', 0)
            pitch = clef_position_to_midi(relative_pos, clef_type)
            duration_value = note_data.get('attributed_duration', 4)
            duration = 4.0 / duration_value if duration_value > 0 else 1.0

            # Créer la note
            note = Note(
                pitch=pitch,
                start_time=current_time,
                duration=duration,
                velocity=80
            )
            treble_measure.add_note(note)
            pos = note_data['order_index']

        treble_staff.add_measure(treble_measure)
        score.add_staff(treble_staff)

    # Portée en clef de fa
    if len(bass_notes) > 0:
        bass_staff = Staff("bass")
        bass_measure = Measure(start_time=0)

        current_time = 0.0
        duration = 0
        pos = 0
        for i, note_data in enumerate(bass_notes):
            current_time += duration if note_data.get('order_index', 0) != pos else 0
            # Calculer la hauteur
            clef_type = note_data.get('attributed_key', 'clefF')
            relative_pos = note_data.get('relative_position', 0)
            pitch = clef_position_to_midi(relative_pos, clef_type)
            duration_value = note_data.get('attributed_duration', 4)
            duration = 4.0 / duration_value if duration_value > 0 else 1.0

            # Créer la note
            note = Note(
                pitch=pitch,
                start_time=current_time,
                duration=duration,
                velocity=80
            )
            bass_measure.add_note(note)
            pos = note_data['order_index']

        bass_staff.add_measure(bass_measure)
        score.add_staff(bass_staff)

    # Convertir en MIDI
    print(f"Conversion en MIDI: {output_file}")
    convert_score_to_midi(score, output_file)
    print(f"Fichier MIDI créé: {output_file}")

    return score

if __name__ == "__main__":
    json_file = "response_1703.json"
    output_file = "super_midi.mid"

    try:
        score = create_midi(json_file, output_file)
        print("Conversion réussie!")

    except Exception as e:
        print(f"Erreur: {str(e)}")
