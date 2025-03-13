"""
Conversion d'une représentation musicale structurée en MIDI.
Code nécessite music_structure.py [à faire, mais dépendra bcp du résultat du YOLO]

Score = classe la plus importante dans music_structure, correspond à la partition
--> tempo, # portées, global_TimeSignature, KeySignature
Faudra aussi créer classes portées et mesures pour pouvoir mettre les notes dans les mesures
--> portée : liste des mesures
--> mesure : time de lancement de la mesure + liste des notes + TimeSignature individuelle au besoin

Note = classe qui correspond à chaque note individuelle
--> hauteur, time, durée, vélocité MIDI (mettre un truc standard), accentuation

TimeSignature = permet de définir la métrique
plus simple de gérer ça dans post-processing
--> numérateur, dénominateur

KeySignature = permet de définir la tonalité
--> # dièses, qui peut également être négatif (si tonalité avec bémols)

"""

import mido

# MUSIC_STRUCTURE A FAIRE
from music_structure import Score, Note, TimeSignature, KeySignature


class AccidentalHandler:
    """
    Gestion des accentuations : dieses, bemols, becarres
    """

    def __init__(self, key_signature: KeySignature = None):
        self.key_signature = key_signature
        self.accidental_map = {
            "sharp": 1,
            "flat": -1,
            "natural": 0,
            "double_sharp": 2,
            "double_flat": -2
        }

        # Détermination de quelles notes sont dieses / bemols
        self.sharp_order = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
        self.flat_order = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

        # bemols/dieses dans la measure
        self.measure_accidentals = {}

    def reset_measure(self):
        self.measure_accidentals = {}

    def get_pitch_adjustment(self, note_name: str):
        """
        Application dieses / bemols liés à la tonalité
        """
        if self.key_signature is None:
            return 0

        # Verification des  accentuations
        sharps = self.key_signature.sharps
        if sharps > 0:
            # En cas de dieses
            for i in range(sharps):
                if i < len(self.sharp_order) and note_name == self.sharp_order[i]:
                    return 1  # on applique un diese a la note
        elif sharps < 0:
            # En cas de bemols
            for i in range(-sharps):
                if i < len(self.flat_order) and note_name == self.flat_order[i]:
                    return -1  # on applique un bemol a la note

        return 0

    def apply_accidentals(self, note: Note):
        """
        Application des dieses / bemols
        Dépend de la tonalité et des accentuations dans la mesure
        """
        # MIDI: C = 0, C# = 1, D = 2, etc. modulo 12
        midi_note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        base_pitch = note.pitch % 12
        base_note_name = midi_note_names[base_pitch][0]  # Extraction de la lettre

        adjustment = 0

        # Verification accentuations
        if note.accidental:
            adjustment = self.accidental_map.get(note.accidental, 0)
            # Stockage de l'accentuation
            self.measure_accidentals[base_note_name] = adjustment

        # On verifie si l'accentuation a été appliquée à la note précedemment
        elif base_note_name in self.measure_accidentals:
            adjustment = self.measure_accidentals[base_note_name]

        # On vérifie si la tonalité implique une accentuation
        else:
            adjustment = self.get_pitch_adjustment(base_note_name)

        return note.pitch + adjustment


class MIDIConverter:
    """
    Convertit résultat du post-processing en MIDI.
    """

    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = ticks_per_beat

    def beats_to_ticks(self, beats: float):
        return round(beats * self.ticks_per_beat)

    def create_midi_file(self, score: Score, output_path: str = None):
        midi = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)

        # Creation MidiTrack
        for i, staff in enumerate(score.staves):
            track = mido.MidiTrack()
            midi.tracks.append(track)

            # Track name
            track.append(mido.MetaMessage('track_name', name=f'Staff {i+1}', time=0))

            # Instrument - piano par defaut
            track.append(mido.Message('program_change', program=0, time=0))

            # Tempo
            tempo_in_microseconds = mido.bpm2tempo(score.tempo)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo_in_microseconds, time=0))

            time_sig = None
            if score.global_time_signature:
                time_sig = score.global_time_signature
            else:
                for measure in staff.measures:
                    if measure.time_signature:
                        time_sig = measure.time_signature
                        break

            if time_sig:
                track.append(mido.MetaMessage(
                    'time_signature',
                    numerator=time_sig.numerator,
                    denominator=time_sig.denominator,
                    clocks_per_click=24,
                    notated_32nd_notes_per_beat=8,
                    time=0
                ))

            accidental_handler = AccidentalHandler(score.key_signature)

            # Ajout de toutes les notes
            for measure in staff.measures:
                # Accentuations remises à 0
                accidental_handler.reset_measure()

                # Tri des notes
                sorted_notes = sorted(measure.notes, key=lambda n: n.start_time)

                # Conversion en MIDI
                last_time = measure.start_time
                for note in sorted_notes:
                    # Application dieses / bemols
                    adjusted_pitch = accidental_handler.apply_accidentals(note)

                    # Evenement Note on
                    delta_time = self.beats_to_ticks(note.start_time - last_time)
                    track.append(mido.Message(
                        'note_on',
                        note=adjusted_pitch,
                        velocity=note.velocity,
                        time=delta_time
                    ))

                    # Update de last time
                    last_time = note.start_time

                    # Evenement Note off
                    delta_time = self.beats_to_ticks(note.duration)
                    track.append(mido.Message(
                        'note_off',
                        note=adjusted_pitch,
                        velocity=0,
                        time=delta_time
                    ))

                    # Update de last time
                    last_time = note.start_time + note.duration

        # Sauvegarde du fichier
        if output_path:
            midi.save(output_path)

        return midi


def convert_score_to_midi(score: Score, output_path: str):
    """ Convertisseur MIDI qui permet de créer une classe convertisseur puis de créer un MIDI"""
    converter = MIDIConverter()
    midi = converter.create_midi_file(score, output_path)
    print(f"MIDI file saved to {output_path}")
    return midi
