"""
Structures de données pour représenter les éléments musicaux.
Ces classes sont utilisées dans midi_converter.
"""

class TimeSignature:
    """Représente une signature rythmique (ex: 4/4, 3/4, etc.)"""

    def __init__(self, numerator, denominator, start_time):
        """
        numerator: Le numérateur (ex: 4 pour 4/4)
        denominator: Le dénominateur (ex: 4 pour 4/4)
        start_time: Le temps de début en beats
        """
        self.numerator = numerator
        self.denominator = denominator
        self.start_time = start_time


class KeySignature:
    """Représente une armure (nombre de dièses ou bémols)"""

    def __init__(self, sharps, start_time):
        """
        sharps: Nombre de dièses (positif) ou bémols (négatif)
        start_time: Le temps de début en beats
        """
        self.sharps = sharps  # Positif pour les dièses, négatif pour les bémols
        self.start_time = start_time


class Note:
    """Représente une note individuelle avec ses propriétés"""

    def __init__(self, pitch, start_time, duration, velocity=64, accidental=None):
        """
        pitch: Hauteur MIDI (0-127)
        start_time: Temps de début en beats
        duration: Durée en beats
        velocity: Vélocité MIDI (0-127), défaut 64
        accidental: dièse, bémol, bécarre
        """
        self.pitch = pitch
        self.start_time = start_time
        self.duration = duration
        self.velocity = velocity
        self.accidental = accidental


class Measure:
    """Représente une mesure contenant des notes"""

    def __init__(self, start_time, notes=None, time_signature=None):
        """
        start_time: Temps de début en beats
        notes: Liste de notes (optionnel, on peut ajouter des notes avec add_notes)
        time_signature: Signature rythmique (optionnel, plutôt géré au global)
        """
        self.start_time = start_time
        self.notes = notes if notes is not None else []
        self.time_signature = time_signature

    def add_note(self, note):
        """
        Ajoute une note à la mesure.
        """
        self.notes.append(note)

class Staff:
    """Représente une portée contenant des mesures"""

    def __init__(self, staff_type, measures=None):
        """
        staff_type: Type de portée ('treble', 'bass', etc.)
        measures: Liste de mesures (optionnel, on eut ajouter des mesures avec add_measures)
        """
        self.staff_type = staff_type
        self.measures = measures if measures is not None else []

    def add_measure(self, measure):
        """
        measure: Mesure à ajouter
        """
        self.measures.append(measure)


class Score:
    """Représente une partition complète"""

    def __init__(self, tempo=120, staves=None, global_time_signature=None, key_signature=None):
        """
        tempo: Tempo en BPM (défaut 120)
        staves: Liste de portées (optionnel, même chose que pour Note et Measure)
        global_time_signature: Signature rythmique globale (objet TimeSignature)
        key_signature: Armure globale (objet KeySignature)
        """
        self.tempo = tempo
        self.staves = staves if staves is not None else []
        self.global_time_signature = global_time_signature
        self.key_signature = key_signature

    def add_staff(self, staff):
        """
        staff: Portée à ajouter
        """
        self.staves.append(staff)
