"""Materialize author-curated contract examples; never claim independent gold."""
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / 'packages/training/data'
# Surface, root, pattern, POS. Inflection/ambiguity is intentionally partial here.
stems = [
 ('كتاب','كتب','فعال','noun'), ('كاتب','كتب','فاعل','noun'),
 ('مكتوب','كتب','مفعول','noun'), ('مدرسة','درس','مفعلة','noun'),
 ('طالب','طلب','فاعل','noun'), ('علم','علم',None,'noun'),
 ('قلم','قلم','فعل','noun'), ('باب','بوب',None,'noun'),
 ('بيت','بيت','فعل','noun'), ('شمس','شمس','فعل','noun'),
 ('قمر','قمر','فعل','noun'), ('رجل','رجل','فعل','noun'),
 ('امرأة',None,None,'noun'), ('طفل','طفل','فعل','noun'),
 ('طريق','طرق','فعيل','noun'), ('مدينة','مدن',None,'noun'),
 ('جميل','جمل','فعيل','adjective'), ('كبير','كبر','فعيل','adjective'),
 ('صغير','صغر','فعيل','adjective'), ('سريع','سرع','فعيل','adjective'),
 ('كتب','كتب','فعل','verb'), ('درس','درس','فعل','verb'),
 ('ذهب','ذهب','فعل','verb'), ('قرأ','قرأ','فعل','verb'),
 ('جلس','جلس','فعل','verb'), ('دخل','دخل','فعل','verb'),
 ('خرج','خرج','فعل','verb'), ('عمل','عمل','فعل','verb'),
 ('فتح','فتح','فعل','verb'), ('سمع','سمع','فعل','verb'),
]
rows = []
def add(parts, root, pattern, pos, lemma, slices=()):
    offset=0; spans=[]
    for surface, kind in parts:
        spans.append({'type':kind,'start':offset,'end':offset+len(surface)})
        offset += len(surface)
    rows.append({'id':f'contract-{len(rows)+1:03}', 'word':''.join(x[0] for x in parts),
      'lemmaFamily':lemma,'labelOrigin':'author-curated', 'reviewStatus':'pending-human-review',
      'slices':list(slices), 'analyses':[{'spans':spans,'root':root,'pattern':pattern,'pos':pos,'features':{},'score':0}]})
for word, root, pattern, pos in stems: add([(word,'stem')],root,pattern,pos,word)
for word in ['كتاب','قلم','بيت','شمس','قمر','طالب','رجل','طفل','طريق','مدينة']:
    _, root, pattern, pos=next(s for s in stems if s[0]==word)
    add([('ال','article'),(word,'stem')],root,pattern,pos,word,['clitic-stack'])
for parts, lemma in [
 ([('و','conjunction'),('ب','preposition'),('كتاب','stem'),('هم','pronominal_enclitic')],'كتاب'),
 ([('ف','conjunction'),('ب','preposition'),('ال','article'),('قلم','stem')],'قلم'),
 ([('و','conjunction'),('ال','article'),('بيت','stem')],'بيت'),
 ([('ب','preposition'),('ال','article'),('مدرسة','stem')],'مدرسة'),
 ([('كتاب','stem'),('ه','pronominal_enclitic')],'كتاب'),
 ([('قلم','stem'),('ي','pronominal_enclitic')],'قلم'),
 ([('و','conjunction'),('كاتب','stem')],'كاتب'),
 ([('ف','conjunction'),('خرج','stem')],'خرج'),
 ([('كتب','stem'),('نا','inflectional_suffix')],'كتب'),
 ([('طالب','stem'),('ات','inflectional_suffix')],'طالب'),
]:
    _,root,pattern,pos=next(s for s in stems if s[0]==lemma)
    add(parts,root,pattern,pos,lemma,['clitic-stack'])
assert len(rows)==50
payload=json.dumps(rows,ensure_ascii=False,indent=2)+'\n'
(out/'contract-fixtures.json').write_text(payload)
(out/'contract-fixtures.sha256').write_text(hashlib.sha256(payload.encode()).hexdigest()+'\n')
