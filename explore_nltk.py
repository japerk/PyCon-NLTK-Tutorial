import collections, itertools, string
from nltk import collocations, probability, stem
from nltk.corpus import stopwords, wordnet
from nltk.tag.util import untag
from nltk.util import bigrams

stopset = set(stopwords.words('english')) | set(string.punctuation)
stemmer = stem.PorterStemmer()

def count_stems(corpus):
	fd = probability.FreqDist()
	
	for word in corpus.words():
		w = word.lower()
		if w in stopset: continue
		fd.inc(stemmer.stem(w))
	
	return fd

def count_hypernyms(corpus):
	fd = probability.FreqDist()
	
	for word in corpus.words():
		w = word.lower()
		if w in stopset: continue
		
		for syn in wordnet.synsets(w):
			if syn.pos != 'n': continue
			
			for path in syn.hypernym_paths():
				for hyp in path:
					fd.inc(hyp.name)
	
	return fd

def count_stemmed_bigram_collocations(corpus, min_freq=3):
	stems = (stemmer.stem(w.lower()) for w in corpus.words())
	finder = collocations.BigramCollocationFinder.from_words(stems)
	finder.apply_word_filter(lambda w: w in stopset)
	finder.apply_freq_filter(min_freq)
	return finder

def count_tag_words(corpus, tagger):
	cfd = probability.ConditionalFreqDist()
	
	for sent in corpus.sents():
		for word, tag in tagger.tag(sent):
			w = word.lower()
			if w in stopset: continue
			cfd[tag].inc(w)
	
	return cfd

def count_phrases(corpus, tagger, chunker):
	cfd = probability.ConditionalFreqDist()
	
	for sent in corpus.sents():
		tree = chunker.parse(tagger.tag(sent))
		
		for sub in tree.subtrees():
			if sub.node == 'S': continue
			words = untag(sub.leaves())
			if len(words) >= 2: cfd[sub.node].inc(' '.join(words))
	
	return cfd

def classify_paras(paras, classifier):
	d = collections.defaultdict(list)
	
	for para in paras:
		words = [w.lower() for w in itertools.chain(*para)]
		feats = dict([(w, True) for w in words + bigrams(words)])
		label = classifier.classify(feats)
		d[label].append(' '.join(words))
	
	return d