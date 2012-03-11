from nltk.chunk.named_entity import NEChunkParser, NEChunkParserTagger
from nltk.classify import NaiveBayesClassifier
from nltk.tag.simplify import simplify_wsj_tag
from nltk.tree import Tree

def simplify_chunk(chunk):
	if isinstance(chunk, Tree):
		return Tree(chunk.node, [simplify_chunk(c) for c in chunk])
	elif isinstance(chunk, tuple):
		word, tag = chunk
		return (word, simplify_wsj_tag(tag))
	else:
		return chunk

# custom classes are required to use a custom classifier, the default is megam

class ChunkTagger(NEChunkParserTagger):
	def _classifier_builder(self, train):
		return NaiveBayesClassifier.train(train)

class ChunkParser(NEChunkParser):
	def _train(self, corpus):
		self._tagger = ChunkTagger([self._parse_to_tagged(s) for s in corpus])