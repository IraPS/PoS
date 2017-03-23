import re

def create_train_markup(filename):
    text = open(filename, 'r', encoding='utf-8')
    words = [(line.split('/')[0], re.findall('^[\w-]+', line.split('/')[2])[0]) for line in text if line != '\n']
    return words

def create_text_to_pars_S(filename):
    text = open(filename, 'r', encoding='utf-8').read().split('\n\n')
    sentences = ''
    for sentence in text:
        for word in sentence.split('\n'):
            sentences += word.split('/')[0] + ' '
        sentences += '.\n'
    sentences = sentences.replace(' .', '.').split('\n')
    x = open('text_to_pars_S1.txt', 'w', encoding='utf-8')
    for sentence in sentences[:15000]:
        x.write(sentence + '\n')
    x.close()
    x = open('text_to_pars_S2.txt', 'w', encoding='utf-8')
    for sentence in sentences[15000:30000]:
        x.write(sentence + '\n')
    x.close()
    x = open('text_to_pars_S3.txt', 'w', encoding='utf-8')
    for sentence in sentences[30000:45000]:
        x.write(sentence + '\n')
    x.close()
    x = open('text_to_pars_S4.txt', 'w', encoding='utf-8')
    for sentence in sentences[45000:60000]:
        x.write(sentence + '\n')
    x.close()
    x = open('text_to_pars_S5.txt', 'w', encoding='utf-8')
    for sentence in sentences[60000:75000]:
        x.write(sentence + '\n')
    x.close()
    x = open('text_to_pars_S7.txt', 'w', encoding='utf-8')
    for sentence in sentences[75000:]:
        x.write(sentence + '\n')
    x.close()
    
def create_text_to_pars(filename):
    text = open(filename, 'r', encoding='utf-8').read().split('\n\n')
    x = open('text_to_pars.txt', 'w', encoding='utf-8')
    for word in create_train_markup(filename):
        x.write(word[0] + '\n')
    x.close()



a = create_train_markup('ruscorpora.parsed.txt')
x = open('lables.txt', 'w', encoding='utf-8')
for i in a[:20000]:
    if re.search('[A-z]', i[0]) is None and re.search('\d', i[0]) is None and '-' not in i[0]:
        x.write(str(i) + '\n')
x.close()
#b = create_text_to_pars_S('ruscorpora.parsed.txt')
