import nltk
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import MWETokenizer
from string import punctuation
import csv
import pymorphy2
import gensim
import re
import json
import unicodedata
from wiktionaryparser import WiktionaryParser
parser = WiktionaryParser()
from copy import deepcopy

rus_stop = stopwords.words('russian') #всякие важные переменные
digits = [str(i) for i in range(1000)]
tok = tokenize.MWETokenizer(separator=' ')
#toknorm = tokenize.WordPunctTokenizer()
morph = pymorphy2.MorphAnalyzer()

text = []
roles = []
dlg = []
with open('test_data.csv', newline='') as data:
    reader = csv.DictReader(data)
    for row in reader:
        text.append(row['text']) # заливаю все реплики в один список
        roles.append(row['role'])
        dlg.append(row['dlg_id'])

for phrase in range(len(text)):
    text[phrase] = text[phrase].split(' ') #разделила фразы в тексте на списки
    for word in text[phrase]:
        if word in (rus_stop): #отсекаю цифры и стоп-слова
            text[phrase].remove(word)
orig = deepcopy(text)
orig_trig = deepcopy(text)

for phrase in range(len(text)):
    text[phrase] = ' '.join(text[phrase])
prog = re.compile('[А-Яа-я]+') # некоторые выражения употребляются комплексно, ищем биграммы
text_for_bigram = (" ").join(text)
bigramm = list(nltk.bigrams(prog.findall(str(text_for_bigram).lower())))
bgfd = nltk.FreqDist(bigramm)
#bgfd.most_common(100)[1][0]
#bgfd.most_common(100)[0][0][0]+ ' ' + bgfd.most_common(100)[0][0][1]
#bigramm
bgfd.most_common(100)
for i in range(len(bgfd.most_common(100))):
    tok.add_mwe(bgfd.most_common(100)[i][0]) #заливаем словосочитания в словарь

for phrase in range(len(text)):
    tok_phrase_norm = nltk.word_tokenize((text[phrase].lower()))
    text[phrase] = tok.tokenize(tok_phrase_norm)

 # поиск приветствий
hi_wiki = parser.fetch('здравствуй','russian') # для того чтобы найти приветствия смотрим синонимы в словаре
hi_syn_emph = hi_wiki[0]['definitions'][0]['examples'][0]
hi_syn_norm = hi_syn_emph.encode('utf-8').replace(b'\xcc\x81', b'').decode('utf-8') #синонимы имеют ударения, убираем их
hi_syn_norm.encode('utf-8')
hi_syn_norm_list = hi_syn_norm.split()
hi_syn_norm_list.append("добрый день");hi_syn_norm_list.append('здравствуй')
hi_list_manager_phrase = []
print("a) Менеджер поздоровался в следующих фразах:")
for phrases in range(len(text)):
    #print(text[phrases])
    for word in range(len(text[phrases])):
        if (text[phrases][word].lower() in hi_syn_norm_list) and roles[phrases] == 'manager':
            #print(text[phrases][word].lower())
            if phrases not in hi_list_manager_phrase:
                hi_list_manager_phrase.append(phrases)
        words = text[phrases][word].split(" ")        
        for subword in words:
            if (subword.lower() in hi_syn_norm_list) and roles[phrases] == 'manager':
                if phrases not in hi_list_manager_phrase:
                    hi_list_manager_phrase.append(phrases)
hi_list_manager_phrase.sort(key=None)
for i in range(len(hi_list_manager_phrase)):
    print(f"Диалог {dlg[hi_list_manager_phrase[i]]}\n{' '.join(text[hi_list_manager_phrase[i]])}")

prob_thresh = 0.4
dlg_name = []
dlg_num = []
print("b) Менеджер назвал свое имя в следующих фразах:")
for or_phrase in range(len(orig)):
        for or_words in range(len(orig[or_phrase])):
            for p in morph.parse((orig[or_phrase][or_words])):
                if 'Name' in p.tag and p.score >= prob_thresh:
                    if (("зовут" in orig[or_phrase]) or ("это" in orig[or_phrase])) and (roles[or_phrase] == "manager"):
                        print(f"Диалог {dlg[or_phrase]}\n{(' ').join(orig[or_phrase])}")
            g = morph.parse(orig[or_phrase][or_words])[0]
            if 'Name' in g.tag and g.score >= prob_thresh and (roles[or_phrase] == "manager") and (("зовут" in orig[or_phrase]) or ("это" in orig[or_phrase])):
                dlg_name.append(orig[or_phrase][or_words])
                dlg_num.append(dlg[or_phrase])
print("\nc) Имя менеджера:")
for i in range(len(dlg_name)):
    print(f"Диалог {dlg_num[i]}\n {dlg_name[i]}")

trigram = list(nltk.trigrams(prog.findall(str(text).lower()))) #тк компания может состоять из нескольких слов
tgfd = nltk.FreqDist(trigram)
print("d) В диалогах упоминались компании:")
for i in range(len(tgfd.most_common(100))):
    if "компания" in tgfd.most_common(100)[i][0][0]:
        print(tgfd.most_common(100)[i][0][1],tgfd.most_common(100)[i][0][2])
for i in range(len(bgfd.most_common(1000000))):
    if "компания" in bgfd.most_common(1000000)[i][0][0]:
        #print(bgfd.most_common(1000000)[i][0])
        g = morph.parse(bgfd.most_common(1000000)[i][0][1])[0]
        if 'NOUN' in g.tag and g.score >= prob_thresh:
            print(bgfd.most_common(1000000)[i][0][1])


bye_wiki = parser.fetch('до свидания','russian') # для того чтобы найти приветствия смотрим синонимы в словаре
bye_syn_emph = bye_wiki[0]['definitions'][0]['examples'][0]
bye_syn_norm = bye_syn_emph.encode('utf-8').replace(b'\xcc\x81', b'').decode('utf-8') #синонимы имеют ударения, убираем их
bye_syn_norm.encode('utf-8')
bye_syn_norm_list = bye_syn_norm.split()
bye_syn_norm_list.remove('пока');bye_syn_norm_list.remove('до')
bye_syn_norm_list.append("до свидания");bye_syn_norm_list.append("свидания");bye_syn_norm_list.append("встречи");bye_syn_norm_list.append("до встречи")
#print(hi_syn_norm_list)
bye_list_manager_phrase = []
print("e) Менеджер попрощался в следующих фразах:")
for phrases in range(len(text)):
    for word in range(len(text[phrases])):
        if (text[phrases][word].lower() in bye_syn_norm_list) and roles[phrases] == 'manager':
            #print(text[phrases][word].lower())
            if phrases not in bye_list_manager_phrase:
                bye_list_manager_phrase.append(phrases)
        words = text[phrases][word].split(" ")        
        for subword in words:
            if (subword.lower() in bye_syn_norm_list) and roles[phrases] == 'manager':
                if phrases not in bye_list_manager_phrase:
                    bye_list_manager_phrase.append(phrases)
bye_list_manager_phrase.sort(key=None)
#print(bye_list_manager_phrase)
for i in range(len(bye_list_manager_phrase)):
    print(f"Диалог {dlg[bye_list_manager_phrase[i]]}\n{' '.join(orig[bye_list_manager_phrase[i]])}")

print("f) Менеджер поздоровался и попрощался:")
dlg_bye = []
for i in range(len(bye_list_manager_phrase)):
    dlg_bye.append(dlg[bye_list_manager_phrase[i]])
for hi in range(len(hi_list_manager_phrase)):
    if dlg[hi_list_manager_phrase[hi]] in dlg_bye:
        print(f"В диалоге {dlg[hi_list_manager_phrase[hi]]}")

