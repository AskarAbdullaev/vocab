import os
import sys
from unicodedata import normalize
import regex as re
from wordfreq import top_n_list, available_languages, tokenize
import requests
from tqdm import tqdm
import spacy
import spellchecker
from bs4 import BeautifulSoup


def analyse_vocab(source: str,
                   language: str = 'en',
                   boundaries: tuple = (10_000, 100_000),
                   model_type: str = 'md',
                   lemmatization: bool = True,
                   spellchecking: bool = True,
                   show_not_found: bool = False):
    """
    Dependencies:
        python=3.12.2
        unicodedata=15.0.0
        regex=2.5.135
        wordfreq=3.7.0
        tqdm=4.66.4
        spacy=3.7.6
        spellchecker=0.8.0
        bs4=4.12.3

    This function takes a text (str) or a URL (in this case, it extracts the text from html using requests
    and BeautifulSoup modules). The raw text is lightly cleaned and tokenized in a primitive way.
    Spacy model for the provided language is loaded (or downloaded, if not already). Model is chosen to be
    compatible with the currently installed spacy version. By default, a middle size model is used, which can be
    changed by setting one of the common model types from ['sm', 'md', 'lg', 'trf'].

    Each unique token is further processed to obtain its corrected version (using pyspellchecker module,
    if required) and its lemma (using previously loaded spacy model). With the help of wordfreq module utility 'top_n_list',
    its corresponding position in a frequency list is obtained for each token. In case when token and its lemma are
    not the same, the smaller frequency index is chosen. If both token and lemma are not found in the frequency list,
    the score is set to -1. Additionally, comments can be attached to such tokens which could not be corrected
    using pyspellchecker ([ Spelling Mistake? ]) or not found in the frequency list ([ Not Found ]).

    There is also a possibility to intentionally exclude certain tokens from the list by putting them into the file
    'exclude_[language_code].txt'.

    Further the tokens are filtered so that only tokens with the required frequency indices are left (specified by
    'boundaries' keyword). At this stage also tokens with non-alpha symbols (except for hyphens) are removed.

    Finally, each raw token still extant is identified in the original text. The context window around the token
    is created (up to +- 300 symbols). These abstracts are used as examples, while the token in question is
    highlighted with [_token_] synthax.

    All the extant tokens are printed down in descending order of frequency index, following the pattern:

    i) lemma ( raw_token ) --- frequency [ comments ]
    example

    Notice: if lemma and raw_token are the same, a single word is printed.

    Parameters:
        source (str): string or URL
        language (str): 2-letter language code. Defaults to 'en'.
        boundaries (tuple): Lower and upper bounds of frequency scores. Defaults to (10_000, 100_000).
        model_type (str): Spacy model type: sm, md, lg or trf. Defaults to 'md'.
        lemmatization (bool): apply spacy lemmatization. Defaults to True.
        spellchecking (bool): apply spellchecking on tokens. Defaults to True.
        show_not_found (bool): also show Not Found tokens in the end. Defaults to False.

    Returns:
        list: list of tokens (each entry is a list of raw_token, frequency_score, corrected_token, lemma, comments, example)
    """

    # Assertions and extracting text from the source (string or url)
    assert isinstance(source, str), 'Invalid source'
    assert language in available_languages().keys(), f'Unknown language code, available codes are: {available_languages().keys()}'
    assert len(boundaries) == 2 and isinstance(boundaries[0], int) and isinstance(boundaries[1], int), 'Invalid boundaries'
    assert boundaries[1] > boundaries[0], 'Invalid boundaries'
    assert model_type in ['sm', 'md', 'lg', 'trf'], f"Model type can be {['sm', 'md', 'lg', 'trf']}"
    assert isinstance(lemmatization, bool)
    assert isinstance(spellchecking, bool)
    assert isinstance(show_not_found, bool)
    lower_bound, upper_bound = boundaries

    if re.match(r'http.+', source):
        text = requests.get(source).content
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
    else:
        text = source[:]

    # Settings (exclude file, spacy, pyspellchecker, wordfreq)
    if not os.path.isfile(f'exclude_{language}.txt'):
        open(f'exclude_{language}.txt', mode='w')
    word_list = top_n_list(f'{language}', int(1e6))

    if lemmatization:
        spacy_version = '.'.join(spacy.__version__.split('.')[0:2])
        compatibility_url = "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json"
        models = requests.get(compatibility_url).json()['spacy'][spacy_version]
        nlp_, available_models = None, []
        for model in models.keys():
            if model[:2] == language:
                available_models.append(model)
                if f'_{model_type}' in model:
                    nlp_ = model
        if not nlp_ and available_models:
            nlp_ = available_models[0]
        if nlp_:
            if nlp_ not in spacy.cli.info()['pipelines']:
                os.system(f'python -m spacy download {nlp_}')
            nlp_ = spacy.load(nlp_)
        else:
            raise ValueError(f'No spacy models found for this version and language {language}')
        nlp = lambda x: nlp_(x)[0].lemma_
    else:
        nlp = lambda x: x

    if spellchecking:
        spelling = lambda x: spellchecker.SpellChecker(language=language).correction(x)
    else:
        spelling = lambda x: x

    # Initial text processing
    text = '  ' + re.sub(r'\n{1,100}', ' ', text) + '  '
    original_text = text[:]
    text = text.lower()
    text_to_list = re.findall(r'\w+', text)
    text_to_set = set(text_to_list)
    print(f'\nTokens in the text: {len(text_to_list)}, unique tokens: {len(text_to_set)}')
    text_to_set = list(text_to_set)

    # Helper functions
    def freq(word: str, word_list_: list):
        try:
            frequency_ = word_list_.index(word)
        except ValueError:
            frequency_ = -1
        return frequency_

    def lines(text_: str, length_: int) -> list[str]:
        text_ = text_.replace('\n', '')
        text_ = text_.split(" ")
        result, pointer, new_line = [], 0, ''
        next_word = ((word_, len(word_)) for word_ in text_)
        for _ in range(len(text_)):
            word, symbols = next(next_word)
            if pointer + symbols > length_:
                result.append(new_line)
                new_line = word + ' '
            else:
                new_line += (word + ' ')
            pointer = len(new_line)
        if len(new_line) > 0:
            result.append(new_line)
        return result

    # Lemmatization / Spelling
    tokens_info_raw = []
    for i, w in tqdm(enumerate(text_to_set), desc='Lemmatization / Spelling...', total=len(text_to_set)):
        corrected = spelling(tokenize(w, language)[0])
        lemma = nlp(corrected if corrected else w)

        # Taking the smaller frequency number (more frequent token) of raw token and lemma.
        # If both tokens are not found, -1 is assigned
        frequency_lemma = freq(lemma, word_list)
        frequency_token = freq(w, word_list)
        frequency = [f for f in [frequency_token, frequency_lemma] if f > -1]
        frequency = -1 if not frequency else min(frequency)

        # Adding comments
        comment = ''
        if not corrected:
            comment += '[ Spelling Mistake? ]'
        if frequency == -1:
            comment += '[ Not Found ]'
        tokens_info_raw.append((w, frequency, corrected, lemma, comment))

    tokens_info_raw = list(set(tokens_info_raw))

    # Exclusions (removing tokens listed in the txt file)
    exclusions = set(normalize('NFC', open(f'exclude_{language}.txt', mode='r').read()).replace('\n', ' ').split(' '))
    tokens_info_raw = list(filter(lambda x: not ((x[0] in exclusions) or (x[3] in exclusions)), tokens_info_raw))

    # Selection (only leaving tokens which have appropriate frequency numbers)
    tokens_info = []
    frequencies = set()
    for token in tqdm(tokens_info_raw, desc='Processing...'):

        if (lower_bound <= token[1] <= upper_bound and len(token[0]) > 1 and len(token[3]) > 1) or token[4]:

            # Also excluding non-ascii or numerical tokens
            if all(symbol.isalpha() or symbol == '-' for symbol in token[3]):
                if token[1] not in frequencies or token[1] == -1:
                    frequencies.add(token[1])
                    tokens_info.append(token)

    # Sorting by frequency in descending order
    tokens_info = sorted(tokens_info, key=lambda x: x[1], reverse=True)

    # Extracting text abstracts to illustrate the token context
    for i, token in tqdm(enumerate(tokens_info), desc='Creating examples...', total=len(tokens_info)):

        original_token = re.search(r'[^a-z]' + token[0] + r'[^a-z]', text)
        index = original_token.start() + 1
        original_token = original_text[original_token.span()[0]+1:original_token.span()[1]-1]

        left_i, left_shift = index, 0
        right_i, right_shift = index, 0
        while True:
            if text[left_i] == '.' or left_i == 0 or (text[left_i + 2] == ' ' and left_shift > 300):
                break
            left_i -= 1
            left_shift += 1
        while right_i < len(text):
            if text[right_i] == '.' or right_i == len(text) or (text[right_i - 2] == ' ' and right_shift > 300):
                break
            right_i += 1
            right_shift += 1

        example = (original_text[left_i:index][2:] + '[_' + original_token + '_]' +
                   original_text[index + len(token[0]) - 2:right_i][2:] + '.')
        example = example[0].upper() + example[1:]
        tokens_info[i] = [original_token] + list(token)[1:] + [example]

    n_tokens = len([t for t in tokens_info if t[1] > -1])
    not_found = len(tokens_info) - n_tokens
    print(f'Selected tokens: {n_tokens}, not found: {not_found}')

    for i, token in enumerate(tokens_info):

        if not show_not_found and token[1] == -1:
            continue
        if token[1] == -1 and tokens_info[i-1][1] > -1:
            print('\n' + ' NOT FOUND TOKENS '.center(60, '-'))
        lemma_adjusted = token[3].capitalize() if token[0][0].isupper() else token[3]
        print(f'\n{i + 1}) {lemma_adjusted} {"( " + token[0] + " )" if lemma_adjusted != token[0] else ""} '
              f'--- {token[1]} {token[4] if token[4] else ""} \n', '\n'.join(lines(token[-1], length_=100)))

    return tokens_info

if __name__ == '__main__':

    analyse_vocab('https://www.grimmstories.com/de/grimm_maerchen/der_alte_sultan',
                   language='de',
                   boundaries=(3000, 20000),
                   spellchecking=False,
                   show_not_found=False)

    analyse_vocab('https://planet-vie.ens.fr/thematiques/cellules-et-molecules/membranes/les-membranes-biologiques-des-structures-dynamiques',
                   language='fr',
                   boundaries=(5000, 20000),
                   spellchecking=False,
                   show_not_found=False)

    analyse_vocab('''Neuro-linguistic programming (NLP) is a pseudoscientific approach to communication, personal
    development, and psychotherapy that first appeared in Richard Bandler and John Grinder's book The Structure of Magic I
    (1975). NLP asserts a connection between neurological processes, language, and acquired behavioral patterns, and that
    these can be changed to achieve specific goals in life.[1][2] According to Bandler and Grinder, NLP can treat problems
    such as phobias, depression, tic disorders, psychosomatic illnesses, near-sightedness,[a] allergy, the common cold,[a]
    and learning disorders,[3][4] often in a single session. They also say that NLP can model the skills of exceptional
    people, allowing anyone to acquire them. Die Fremdwörter.''',
                   language='en',
                   boundaries=(10000, 100000),
                   spellchecking=True,
                   show_not_found=True)
