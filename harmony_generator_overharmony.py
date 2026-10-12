#!/usr/bin/env python3
"""
Four-voice harmony generator.

Given a root note (e.g. "C4"), this script finds every combination of four
notes drawn from the chromatic scale within a two-octave window centered on
the root note -- one octave below to one octave above (for C4, that window
is C3 to C5). The root note is included as one of the four voices by
default.

Rules (both apply to every pair of *neighboring* voices, i.e. the chord
sorted low to high):
  1. No two neighboring notes may be within 18% of each other in
     frequency -- each note must be at least 18% higher than the note
     directly below it.
  2. No two neighboring notes may be more than a perfect 5th apart
     (7 semitones) -- voices must stay close enough together to sound
     like a single chord rather than spread across octaves.

Optionally, --upper-resonance adds two extra fixed notes above each
harmony, based on the root note only (not on the rest of the chord),
turning each four-voice harmony into a six-note chord:
  - root + an octave + a perfect 5th (root + 19 semitones)
  - a perfect 4th above THAT note, which lands exactly two octaves above
    the root (root + 24 semitones), since a 5th plus a 4th makes an
    octave

Separately, --jazz-it-up adds the classic jazz 9-11-13 upper extensions
above each harmony, also based on the root note only:
  - 9th:  root + octave + major 2nd  (root + 14 semitones)
  - 11th: root + octave + perfect 4th (root + 17 semitones)
  - 13th: root + octave + major 6th  (root + 21 semitones)
The two flags are independent and can be combined.

Usage:
    python3 harmony_generator.py C4
    python3 harmony_generator.py A#3 --limit 10
    python3 harmony_generator.py Eb4 --min-percent 25
    python3 harmony_generator.py C4 --any-four
    python3 harmony_generator.py C4 --upper-resonance --limit 20
    python3 harmony_generator.py C4 --jazz-it-up --limit 20
    python3 harmony_generator.py C4 --upper-resonance --jazz-it-up --limit 20
    python3 harmony_generator.py C4 --limit 20 --export-musicxml harmonies.musicxml
"""

import argparse
import itertools
import re
import xml.sax.saxutils as saxutils

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Flat spellings map onto the same pitch classes as the sharp names above.
FLAT_TO_SHARP = {
    'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#',
    'CB': 'B', 'FB': 'E',
}

A4_FREQ = 440.0
A4_MIDI = 69  # Scientific pitch notation: C4 = MIDI 60, A4 = MIDI 69


def note_to_midi(note: str) -> int:
    """Convert a note name like 'C4', 'A#3', 'Db4' to a MIDI note number."""
    m = re.match(r'^([A-Ga-g])([#b]?)(-?\d+)$', note.strip())
    if not m:
        raise ValueError(f"Could not parse note name: {note!r}")
    letter, accidental, octave = m.groups()
    letter = letter.upper()
    octave = int(octave)

    if accidental == 'b':
        pitch_name = FLAT_TO_SHARP.get((letter + 'B'), letter)
    else:
        pitch_name = letter + accidental  # '' or '#'

    if pitch_name not in NOTE_NAMES:
        raise ValueError(f"Unknown pitch class in note: {note!r}")

    pitch_class = NOTE_NAMES.index(pitch_name)
    midi = (octave + 1) * 12 + pitch_class
    return midi


def midi_to_freq(midi: int) -> float:
    return A4_FREQ * (2 ** ((midi - A4_MIDI) / 12))


def midi_to_note(midi: int) -> str:
    pitch_class = midi % 12
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[pitch_class]}{octave}"


def generate_chromatic_range(root_midi: int):
    """All chromatic MIDI notes from one octave below to one octave above root."""
    return list(range(root_midi - 12, root_midi + 12 + 1))


def semitones_to_ratio(semitones: float) -> float:
    """Convert an interval in semitones to a frequency ratio (equal temperament)."""
    return 2 ** (semitones / 12)


def satisfies_interval_rules(freqs, min_ratio=1.18, max_ratio=None):
    """True if every pair of *neighboring* sorted frequencies (a) differs by
    at least min_ratio and (b) -- if max_ratio is given -- differs by no
    more than max_ratio. Checking neighbors only is sufficient for the
    lower bound (ratios compound going outward), and is exactly what we
    want for the upper bound too: it's adjacent voices that must stay
    within a 5th of each other, not the outer span of the whole chord."""
    sorted_freqs = sorted(freqs)
    for lo, hi in zip(sorted_freqs, sorted_freqs[1:]):
        ratio = hi / lo
        if ratio < min_ratio:
            return False
        if max_ratio is not None and ratio > max_ratio:
            return False
    return True


THIRD_SEMITONES = (3, 4)  # minor third, major third


def has_stacked_thirds(sorted_combo, third_semitones=THIRD_SEMITONES):
    """True if any two *consecutive* neighboring intervals in the sorted
    chord are both thirds (3 or 4 semitones) -- i.e. the chord contains a
    plain stacked triad like C-E-G (major 3rd + minor 3rd) or C-Eb-Gb
    (minor 3rd + minor 3rd) somewhere inside it. Uses exact semitone
    (MIDI) differences rather than frequency ratios, since thirds are
    exact integer intervals in equal temperament."""
    intervals = [hi - lo for lo, hi in zip(sorted_combo, sorted_combo[1:])]
    return any(
        i1 in third_semitones and i2 in third_semitones
        for i1, i2 in zip(intervals, intervals[1:])
    )


def generate_harmonies(root_note: str, min_ratio: float = 1.18, max_ratio: float = None,
                        require_root: bool = True, allow_stacked_thirds: bool = False):
    root_midi = note_to_midi(root_note)
    candidate_midis = generate_chromatic_range(root_midi)

    if require_root:
        others = [m for m in candidate_midis if m != root_midi]
        combos = ((root_midi,) + combo for combo in itertools.combinations(others, 3))
    else:
        combos = itertools.combinations(candidate_midis, 4)

    harmonies = []
    for combo in combos:
        sorted_combo = tuple(sorted(combo))
        freqs = [midi_to_freq(m) for m in sorted_combo]
        if not satisfies_interval_rules(freqs, min_ratio, max_ratio):
            continue
        if not allow_stacked_thirds and has_stacked_thirds(sorted_combo):
            continue
        harmonies.append(sorted_combo)

    return harmonies


def add_jazz_extensions(combo, root_midi: int):
    """Append the classic jazz '9-11-13' upper extensions, derived only
    from the root note, to a harmony:
      - 9th:  root + 14 semitones (an octave plus a major 2nd)
      - 11th: root + 17 semitones (an octave plus a perfect 4th)
      - 13th: root + 21 semitones (an octave plus a major 6th)
    These are the real jazz-theory extensions (stacked thirds above the
    7th, spelled out here directly in semitones), as opposed to
    --upper-resonance's overtone-series pair."""
    ninth = root_midi + 14
    eleventh = root_midi + 17
    thirteenth = root_midi + 21
    return tuple(sorted(combo + (ninth, eleventh, thirteenth)))


def add_upper_resonance(combo, root_midi: int):
    """Append two fixed 'upper resonance' notes, derived only from the root
    note, to a harmony:
      - root + 19 semitones (an octave plus a perfect 5th above root)
      - root + 24 semitones (a perfect 4th above that note -- exactly two
        octaves above the root, since P5 + P4 = an octave)
    Note on naming: measured strictly by scale-degree count these land on
    the 12th and the 15th (double octave) above the root, not the 13th and
    17th -- but the *intervals themselves* (octave+5th, then +4th) are
    implemented exactly as described, regardless of what they're called."""
    upper_fifth = root_midi + 19
    upper_octave = upper_fifth + 5
    return tuple(sorted(combo + (upper_fifth, upper_octave)))


def format_harmony(combo):
    notes = [midi_to_note(m) for m in combo]
    freqs = [round(midi_to_freq(m), 2) for m in combo]
    return ", ".join(f"{n} ({f} Hz)" for n, f in zip(notes, freqs))


def midi_to_musicxml_pitch(midi: int):
    """Return (step, alter, octave) for a MIDI note, spelled with sharps
    to match NOTE_NAMES (e.g. 61 -> ('C', 1, 4) for C#4)."""
    pitch_class = midi % 12
    octave = midi // 12 - 1
    name = NOTE_NAMES[pitch_class]
    if '#' in name:
        step, alter = name[0], 1
    else:
        step, alter = name, 0
    return step, alter, octave


def note_xml(midi: int, is_chord_member: bool) -> str:
    """Build a single <note> element for a whole note, optionally marked
    as part of a chord (stacked on the previous note in the measure)."""
    step, alter, octave = midi_to_musicxml_pitch(midi)
    alter_line = f"          <alter>{alter}</alter>\n" if alter else ""
    chord_line = "        <chord/>\n" if is_chord_member else ""
    return (
        "      <note>\n"
        + chord_line
        + "        <pitch>\n"
        + f"          <step>{step}</step>\n"
        + alter_line
        + f"          <octave>{octave}</octave>\n"
        + "        </pitch>\n"
        + "        <duration>4</duration>\n"
        + "        <voice>1</voice>\n"
        + "        <type>whole</type>\n"
        + "      </note>\n"
    )


def build_measure_xml(measure_number: int, combo, include_attributes: bool) -> str:
    """One measure holding the whole-note chord for a single harmony combo."""
    parts = [f'    <measure number="{measure_number}">\n']
    if include_attributes:
        parts.append(
            "      <attributes>\n"
            "        <divisions>1</divisions>\n"
            "        <key><fifths>0</fifths></key>\n"
            "        <time><beats>4</beats><beat-type>4</beat-type></time>\n"
            "        <clef><sign>G</sign><line>2</line></clef>\n"
            "      </attributes>\n"
        )
    for i, midi in enumerate(combo):
        parts.append(note_xml(midi, is_chord_member=(i > 0)))
    parts.append("    </measure>\n")
    return "".join(parts)


def build_musicxml(harmonies, root_note: str) -> str:
    """Build a full MusicXML (score-partwise) document with one measure per
    harmony, each measure a single whole-note chord of the four voices."""
    measures = []
    for i, combo in enumerate(harmonies, 1):
        measures.append(build_measure_xml(i, combo, include_attributes=(i == 1)))

    part_name = saxutils.escape(f"Harmonies on {root_note}")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/musicxml.dtd">\n'
        '<score-partwise version="4.0">\n'
        "  <part-list>\n"
        '    <score-part id="P1">\n'
        f"      <part-name>{part_name}</part-name>\n"
        "    </score-part>\n"
        "  </part-list>\n"
        '  <part id="P1">\n'
        + "".join(measures)
        + "  </part>\n"
        "</score-partwise>\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate every valid four-voice harmony around a root note."
    )
    parser.add_argument("root_note", help="Root note, e.g. C4, A#3, Eb5")
    parser.add_argument(
        "--min-percent", type=float, default=18.0,
        help="Minimum percent frequency separation required between neighboring notes (default: 18)"
    )
    parser.add_argument(
        "--max-interval-semitones", type=float, default=7.0,
        help="Largest interval allowed between neighboring voices, in semitones "
             "(default: 7, a perfect 5th). Use 0 or a negative number to disable "
             "this cap."
    )
    parser.add_argument(
        "--allow-stacked-thirds", action="store_true",
        help="Allow chords that contain a plain stacked-triad shape (two "
             "consecutive neighboring intervals that are both thirds, e.g. "
             "C-E-G). Disallowed by default."
    )
    parser.add_argument(
        "--upper-resonance", action="store_true",
        help="Add two extra fixed notes above each harmony, based on the "
             "root note: root + octave + perfect 5th, then a perfect 4th "
             "above that (two octaves above root). Turns each 4-note "
             "harmony into a 6-note chord."
    )
    parser.add_argument(
        "--jazz-it-up", action="store_true",
        help="Add the classic jazz 9-11-13 upper extensions above each "
             "harmony, based on the root note: root + octave + major 2nd "
             "(9th), root + octave + perfect 4th (11th), root + octave + "
             "major 6th (13th). Adds 3 notes per harmony; combine with "
             "--upper-resonance to add both."
    )
    parser.add_argument(
        "--any-four", action="store_true",
        help="Don't require the root note to be one of the four voices; "
             "pick any 4 notes from the two-octave range instead."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only print the first N results, and only export this many if "
             "--export-musicxml is used (default: print/export all)"
    )
    parser.add_argument(
        "--export-musicxml", metavar="PATH",
        help="Write the harmonies to a MusicXML file at PATH, one measure per "
             "harmony, each written as a whole-note chord."
    )
    args = parser.parse_args()

    min_ratio = 1 + (args.min_percent / 100.0)
    max_ratio = (
        semitones_to_ratio(args.max_interval_semitones)
        if args.max_interval_semitones > 0 else None
    )
    root_midi = note_to_midi(args.root_note)
    harmonies = generate_harmonies(
        args.root_note, min_ratio=min_ratio, max_ratio=max_ratio,
        require_root=not args.any_four, allow_stacked_thirds=args.allow_stacked_thirds
    )

    print(f"Root note: {args.root_note}")
    print(f"Range: {midi_to_note(root_midi - 12)} to {midi_to_note(root_midi + 12)}")
    print(f"Minimum separation between neighboring notes: {args.min_percent}%")
    if max_ratio is not None:
        print(f"Maximum interval between neighboring notes: {args.max_interval_semitones} semitones")
    print(f"Stacked-triad shapes (e.g. C-E-G) allowed: {args.allow_stacked_thirds}")
    print(f"Root note required in every harmony: {not args.any_four}")
    if args.upper_resonance:
        fifth, octv = add_upper_resonance((), root_midi)
        print(f"Upper resonance added: {midi_to_note(fifth)} and {midi_to_note(octv)} "
              f"above every harmony (+2 notes)")
    if args.jazz_it_up:
        ninth, eleventh, thirteenth = add_jazz_extensions((), root_midi)
        print(f"Jazz extensions added: {midi_to_note(ninth)} (9th), "
              f"{midi_to_note(eleventh)} (11th), {midi_to_note(thirteenth)} (13th) "
              f"above every harmony (+3 notes)")
    total_notes = 4 + (2 if args.upper_resonance else 0) + (3 if args.jazz_it_up else 0)
    if args.upper_resonance or args.jazz_it_up:
        print(f"Notes per harmony: {total_notes}")
    print(f"Total valid harmonies found: {len(harmonies)}\n")

    to_show = harmonies if args.limit is None else harmonies[:args.limit]
    if args.upper_resonance:
        to_show = [add_upper_resonance(combo, root_midi) for combo in to_show]
    if args.jazz_it_up:
        to_show = [add_jazz_extensions(combo, root_midi) for combo in to_show]

    for i, combo in enumerate(to_show, 1):
        print(f"{i:>5}. {format_harmony(combo)}")

    if args.export_musicxml:
        xml_doc = build_musicxml(to_show, args.root_note)
        with open(args.export_musicxml, "w", encoding="utf-8") as f:
            f.write(xml_doc)
        print(f"\nWrote {len(to_show)} harmonies to {args.export_musicxml} "
              f"(one whole-note chord per measure).")


if __name__ == "__main__":
    main()
