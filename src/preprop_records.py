import pandas as pd
import numpy as np
import spacy
import re
from tqdm import tqdm
tqdm.pandas()
from spellchecker import SpellChecker

spacy.download("en_core_web_trf")

def preprocess_text(text, abb_dict, l_model):
    # Lower-case
    text = text.lower()

    # Remove extra whitespaces
    text = ' '.join(text.split())

    # Replace '&' with 'and'
    text = text.replace(' & ', ' and ')

    # Substitute '/' with '' in the matched patterns
    pattern = re.compile(r'\b([a-zA-Z])\/([a-zA-Z])\b')
    text = re.sub(pattern, r'\1\2', text)

    # Contraction expansions
    text = text.replace('won\'t', 'will not')
    text = text.replace('didn\'t', 'did not')
    text = text.replace('wouldn\'t', 'would not')
    text = text.replace('don\'t', 'do not')
    text = text.replace('couldn\'t', 'could not')
    text = text.replace('we\'ve', 'we have')
    text = text.replace('doesn\'t', 'does not')
    text = text.replace('can\'t', 'cannot')

    # Remove characters after apostrophes
    pattern = re.compile(r"((?:^|\s))(\w{2,3})'\w+(?=\s|$)")
    text = pattern.sub(r'\1\2', text)

    # Remove non-alphabet characters except spaces
    text_l = []
    for ch in text:
        if ch.isalpha() or ch == ' ':
            ch = ch
        else:
            ch = ' '
        text_l.append(ch)
    text = ''.join(text_l)
    
    # manual corrections
    text = text.replace('stand off', 'standoff')
    text = text.replace('oversped', 'overspeed')
    text = text.replace('approx ', 'approximate')
    text = text.replace('annunc ', 'annunciator')
    text = text.replace('aribox', 'airbox')
    text = text.replace('air box', 'airbox')
    text = text.replace('crank shaft', 'crankshaft')
    text = text.replace('crank case', 'crankcase')
    text = text.replace('lcyoming', 'lycoming')

    spell = SpellChecker()
    misspelled = spell.unknown(text.split())
    for word in misspelled:
        if spell.correction(word) is not None and word not in abb_dict.keys() and word not in ('rpm', 'und', 'lycoming', 'overspeed', 'overspeeding', 'helicoil', 'agb', 'retightened', 'nfeather', 'discrep', 'sandell', 'stiffner', 'insp', 'autorotate', 'northwood', 'unsecure', 'rpms', 'rtv', 'adel', 'sniffler', 'resecured', 'comms', 'comm', 'sniffler', 'tanis', 'doublers', 'decel', 'borescope', 'nutplate', 'runup', 'enroute', 'mayville', 'airbox', 'trans', 'cyls', 'kias', 'pre', 'shanked', 'shank', 'helo', 'mins', 'retorqued', 'lueco', 'fosston', 'crookston', 'gapped', 'safetied', 'recowled', 'count', 'airmotive', 'sumped', 'sumping', 'reriveted', 'airboxes', 'swedged', 'swedging', 'debaffled', 'lockplate', 'plio', 'pushrod', 'cherrymax', 'tech', 'reinstallation', 'uncowled', 'retorque', 'unfeather') and len(word) > 3:
            text = text.replace(word, spell.correction(word))
    
    # Abbreviation expansion
    for abbr, expansion in abb_dict.items():
        text = ' '.join([expansion if word == abbr else word for word in text.split()])

    # Stopword removal + lemmatization using spaCy
    text = " ".join(token.lemma_ for token in l_model(text) if not token.is_stop)
    return text


# Load spaCy language model
lm = spacy.load("en_core_web_trf", disable=["parser", "ner"])

# Load abbreviations CSV and create a dictionary
abbreviations_df = pd.read_csv('../data/abbreviations.csv')
abbreviation_dict = dict(zip(abbreviations_df['Abbreviated'], abbreviations_df['Standard_Description']))

# Load grammar CSV and create a custom tags dictionary (if needed later)
grammar_df = pd.read_csv('../data/grammar.csv')
grammar_df = grammar_df[['Word', 'Lemma', 'Part of Speech (POS)']]

pos_map = {
    "NN": "NOUN",
    "ADJ": "ADJ",
    "VB": "VERB",
}

if "attribute_ruler" not in lm.pipe_names:
    lm.add_pipe("attribute_ruler", before="lemmatizer")
ruler = lm.get_pipe("attribute_ruler")

for _, row in grammar_df.iterrows():
    word = row["Word"]
    lemma = row["Lemma"]
    pos_tag = pos_map.get(row['Part of Speech (POS)'], None)
    # build the attrs dict
    attrs = {"LEMMA": lemma}
    if pos_tag:
        attrs["POS"] = pos_tag

    # pattern: match the token by lowercase form
    pattern = [{"LOWER": word}]
    # index=0 so your rules run before any built‑in ones
    ruler.add(patterns=[pattern], attrs=attrs, index=0)

pattern = [{"LOWER": 'left'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'left', "POS": 'NOUN'}, index=0)

pattern = [{"LOWER": 'leaking'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'leak', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'mounting'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'mount', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'cracking'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'crack', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'connecting'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'connect', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'running'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'run', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'roughness'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'rough', "POS": 'ADJ'}, index=0)

pattern = [{"LOWER": 'failed'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'fail', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'cooler'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'cooler', "POS": 'NOUN'}, index=0)

pattern = [{"LOWER": 'worn'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'wear', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'baffling'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'baffle', "POS": 'NOUN'}, index=0)

pattern = [{"LOWER": 'installed'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'install', "POS": 'VERB'}, index=0)

pattern = [{"LOWER": 'filler'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'filler', "POS": 'NOUN'}, index=0)

pattern = [{"LOWER": 'chafing'}]
ruler.add(patterns=[pattern], attrs={"LEMMA": 'chafe', "POS": 'VERB'}, index=0)

lm.vocab['not'].is_stop = False

# Load data
df = pd.read_csv("../data/maintenance_items_2022_grouped.csv") 

# Process the PROBLEM column and save to CSV
df['cleaned_problem'] = df['PROBLEM'].progress_apply(lambda x: preprocess_text(x, abbreviation_dict, lm))
df.to_csv('../data/preprocessed_new_corenlp_grouped.csv', index=False)

# Process the ACTION column and update the same CSV
df['cleaned_action'] = df['ACTION'].progress_apply(lambda x: preprocess_text(x, abbreviation_dict, lm))
df.to_csv('../data/preprocessed_new_corenlp_grouped.csv', index=False)
