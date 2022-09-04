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
    
    #remove empty sentences
    return list(filter(None, sentences))