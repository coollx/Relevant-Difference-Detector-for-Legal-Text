import string
from spellchecker import SpellChecker

#cache the spell checker
spell = SpellChecker()

def elem2sent(article, break_sentence = True):
    '''
    Break article into sentences. Break sentences with "."
    article: xml element
    '''
    #if article is a list of sentences
    sentences = [_ for _ in article.itertext() if not _.isspace() and len(_.split()) > 5]
    
    #break sentences with "."
    if break_sentence:
        sentences = [sent for sub in map(lambda x: x.split('.'), sentences) for sent in sub if len(sent.split()) > 5]
    
    sentences = [sent.strip().lower().translate(str.maketrans('', '', string.punctuation)) for sent in sentences]

    #remove the indices of sentences
    sentences = [sent.split(' ', 1)[1] if len(sent[0]) <= 3 else sent for sent in sentences]

    #correct spellings
    #sentences = [correct_spellings(sent) for sent in sentences]

    #remove empty sentences
    return list(filter(None, sentences))



def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(list(filter(None, corrected_text)))
        