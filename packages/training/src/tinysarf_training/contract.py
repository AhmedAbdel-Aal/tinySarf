"""Versioned normalization and label order shared through an exported manifest."""
import re
LETTERS = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهويى'
MARKS = re.compile('[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]')
SPAN_TYPES = ['conjunction','particle','preposition','article','stem','derivational_suffix','inflectional_suffix','pronominal_enclitic']
SPECIAL = ['__missing__','__unknown__','__na__']
LEGACY_MAPPING = 'legacy-v2'
CORRECTED_MAPPING = 'corrected-v1'
FEATURE_MAP = {
 'per': ('person', {'1':'first','2':'second','3':'third'}),
 'gen': ('gender', {'m':'masculine','f':'feminine'}),
 'num': ('number', {'s':'singular','d':'dual','p':'plural'}),
 'asp': ('aspect', {'p':'perfective','i':'imperfective','c':'imperative'}),
 'mod': ('mood', {'i':'indicative','s':'subjunctive','j':'jussive'}),
 'vox': ('voice', {'a':'active','p':'passive'}),
 'cas': ('case', {'n':'nominative','a':'accusative','g':'genitive'}),
 'stt': ('state', {'d':'definite','i':'indefinite','c':'construct'}),
}
POS_MAP = {'noun':'noun','noun_prop':'proper_noun','noun_num':'numeral','noun_quant':'noun','adj':'adjective','adj_comp':'adjective','adj_num':'adjective','verb':'verb','verb_pseudo':'verb','verb_nom':'noun','adv':'adverb','prep':'preposition','conj':'conjunction','conj_sub':'conjunction','interj':'interjection'}
def normalize(word):
    if not isinstance(word,str) or not word or len(word.encode('utf-16-le'))//2>512 or MARKS.match(word):
        raise ValueError('Expected one Arabic word without leading marks')
    word=MARKS.sub('', word)
    if not word or len(word)>32 or any(c not in LETTERS for c in word):
        raise ValueError('Expected 1–32 Arabic letters')
    return word
def dediac(word): return MARKS.sub('', word)
def lemma_family(value): return re.sub(r'[_\d]+$', '', dediac(value)).replace('-', '')
def internal_feature(analysis, source):
    if source not in analysis: return '__missing__'
    if analysis[source]=='na': return '__na__'
    return FEATURE_MAP[source][1].get(analysis[source], '__unknown__')
def segment_type(tag, version=LEGACY_MAPPING):
    if version not in (LEGACY_MAPPING, CORRECTED_MAPPING): raise ValueError('Unknown label mapping version')
    if version == CORRECTED_MAPPING and tag in {'CONNEC_PART', 'RC_PART'}: return 'particle'
    if tag in {'CONJ','SUB_CONJ'}: return 'conjunction'
    if tag=='PREP': return 'preposition'
    if tag=='DET': return 'article'
    if 'POSS_PRON' in tag or 'DO' in tag or tag=='PRON': return 'pronominal_enclitic'
    if tag.startswith(('NSUFF','PVSUFF','IVSUFF','CVSUFF')): return 'inflectional_suffix'
    if tag.startswith(('PART','FUT','NEG','INTERROG','JUS','SUBJ')): return 'particle'
    return 'stem'

def mapped_pattern(raw, version=LEGACY_MAPPING):
    """Keep the frozen legacy representation unless correction is explicitly requested."""
    if version not in (LEGACY_MAPPING, CORRECTED_MAPPING): raise ValueError('Unknown label mapping version')
    pattern=raw.get('pattern_abstract') or raw.get('pattern')
    if not pattern or pattern in ('na','NOAN','NTWS'): return None
    if version == CORRECTED_MAPPING and raw.get('pattern') in ('na', 'NOAN', 'NTWS'):
        # Some pinned teacher entries transliterate NTWS inside pattern_abstract.
        # An Arabic derivational ending does not supply the missing stem pattern.
        return None
    pattern=pattern.translate(str.maketrans({'1':'ف','2':'ع','3':'ل','4':'ل'}))
    if version == CORRECTED_MAPPING:
        # '+' is the teacher's morpheme separator, not a character in the pattern.
        pattern=pattern.replace('+', '')
        if not any(c in LETTERS or c == 'ٱ' for c in pattern): return None
        if any(c not in LETTERS and c != 'ٱ' and not MARKS.fullmatch(c) for c in pattern): return None
    return pattern

def map_analysis(word, raw, version=LEGACY_MAPPING):
    """Strict surface alignment. Non-concatenative teacher segmentations are logged and dropped."""
    if version not in (LEGACY_MAPPING, CORRECTED_MAPPING): raise ValueError('Unknown label mapping version')
    word=normalize(word); spans=[]; pieces=[]
    for part in raw.get('bw','').split('+'):
        if '/' not in part: continue
        surface, tag=part.rsplit('/',1)
        # The teacher spells hamzat al-wasl explicitly, while its surface index uses alef.
        surface=dediac(surface).replace('ٱ','ا')
        if not surface or surface=='(null)': continue
        pieces.append((surface, segment_type(tag, version)))
    if ''.join(p[0] for p in pieces)!=word:
        raise ValueError('teacher segmentation does not concatenate to normalized surface')
    offset=0
    for surface, kind in pieces:
        end=offset+len(surface)
        if spans and spans[-1]['type']==kind: spans[-1]['end']=end
        else: spans.append({'type':kind,'start':offset,'end':end})
        offset=end
    # Standalone function words are lexical stems in the span contract.
    if not any(s['type']=='stem' for s in spans): spans=[{'type':'stem','start':0,'end':len(word)}]
    root=''.join(c for c in dediac(raw.get('root','')) if c in LETTERS)
    if len(root) not in (3,4): root=None
    pos=raw.get('pos','')
    pos=POS_MAP.get(pos, 'pronoun' if pos.startswith('pron') else 'particle' if pos.startswith('part') else 'unknown')
    pattern=mapped_pattern(raw, version)
    return {'spans':spans,'root':root,'pattern':pattern,'pos':pos,
      'features':{target:internal_feature(raw,source) for source,(target,_) in FEATURE_MAP.items()},
      'lemmaFamily':lemma_family(raw.get('lex',word))}
