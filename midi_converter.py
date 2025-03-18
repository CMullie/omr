import mido

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

            # Traitement des mesures
            for measure in staff.measures:
                # Réinitialiser les altérations au début de chaque mesure
                accidental_handler.reset_measure()

                # Organiser les notes par temps (pour gérer les accords)
                notes_by_time = {}
                for note in measure.notes:
                    start_time = note.start_time
                    if start_time not in notes_by_time:
                        notes_by_time[start_time] = []
                    notes_by_time[start_time].append(note)

                # Trier les temps de début pour traiter les notes dans l'ordre chronologique
                sorted_start_times = sorted(notes_by_time.keys())

                # Variable pour suivre la position temporelle actuelle dans le flux MIDI
                current_time = measure.start_time
                last_end_time = measure.start_time

                # Traiter chaque groupe de notes (accords ou notes individuelles)
                for start_time in sorted_start_times:
                    # Calculer le delta temps depuis la dernière action
                    delta_time = self.beats_to_ticks(start_time - current_time)
                    current_time = start_time

                    # Groupe de notes commençant au même moment (accord)
                    chord_notes = notes_by_time[start_time]

                    # Ajouter tous les événements note_on de l'accord
                    for idx, note in enumerate(chord_notes):
                        # Appliquer les altérations
                        adjusted_pitch = accidental_handler.apply_accidentals(note)

                        # Pour la première note, utiliser le delta_time calculé
                        if idx == 0:
                            track.append(mido.Message(
                                'note_on',
                                note=adjusted_pitch,
                                velocity=note.velocity,
                                time=delta_time
                            ))
                        else:
                            # Pour les autres notes de l'accord, le delta_time est 0
                            track.append(mido.Message(
                                'note_on',
                                note=adjusted_pitch,
                                velocity=note.velocity,
                                time=0
                            ))

                    # Déterminer le temps de fin pour ce groupe (toutes les notes d'un accord ont la même durée)
                    end_time = start_time + chord_notes[0].duration

                    # Mettre à jour pour suivre la fin de l'accord
                    if end_time > last_end_time:
                        last_end_time = end_time

                    # Calculer le delta_time pour les événements note_off
                    note_duration_ticks = self.beats_to_ticks(chord_notes[0].duration)

                    # Ajouter tous les événements note_off de l'accord
                    for idx, note in enumerate(chord_notes):
                        adjusted_pitch = accidental_handler.apply_accidentals(note)

                        # Pour le dernier note_off, mettre à jour current_time
                        if idx == len(chord_notes) - 1:
                            current_time = end_time

                        track.append(mido.Message(
                            'note_off',
                            note=adjusted_pitch,
                            velocity=0,
                            time=note_duration_ticks if idx == 0 else 0  # Delta time seulement pour la première note
                        ))

        # Sauvegarder le fichier MIDI si un chemin est fourni
        if output_path:
            midi.save(output_path)

        return midi

def convert_score_to_midi(score: Score, output_path: str):
    """ Convertisseur MIDI qui permet de créer une classe convertisseur puis de créer un MIDI"""
    converter = MIDIConverter()
    midi = converter.create_midi_file(score, output_path)
    print(f"MIDI file saved to {output_path}")
    return midi
