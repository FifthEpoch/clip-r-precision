# Compositional Split Generation for CLIP-R Precision Evaluations
Implementation of compositional split generation according to [Benchmark for Compositional Text-to-Image Synthesis](https://openreview.net/pdf?id=bKBhQhPeKaF). 

Below is an excerpt on Compositional Split Generation from the Benchmark paper where CLIP-R precision was introduced.

> In order to create the aforementioned compositional splits, we first identify representative sets of
nouns and adjectives present in the dataset, and curate a list of synonyms for each word to account
for the variations in how they manifest in the captions. When selecting adjectives, we first determine
the 60 most frequent adjectives that appear in the dataset. Then we filter out ones that are either color-
related or shape-related. Then, we find 100 most frequent adjective-noun pairs that are associated
with the selected adjectives and extract all the nouns. We use Spacy to tag, lemmatize, and parse
the captions.

> Once the nouns and adjectives are determined, we select the “novel” adjective-noun pairs that will
constitute the evaluation set. Specifically, we calculate the frequencies of all the adjective-noun pairs
and sort them from the most to the least frequent. Then, 10% of the unique adjective-noun pairs are
withheld to become unseen; they are randomly sampled from between the 25th and 75th percentiles
of the sorted list. (Sampling heldout pairs from the top of the list, i.e. most frequent ones, results in
significant shrinkage of the training dataset and limits the variations in nouns present in the evaluation
splits as there exists certain nouns that appear more frequently across pairs, while sampling from
the long-tail results in a small test set.) Finally, based on the withheld pairs the dataset is split into
$D_{train}$, $D_{test seen}$, and $D_{test unseen}$. When generating $D_{test swapped}$ from $D_{test}$ seen, we keep the nouns and
only modify the adjectives so that they form unseen pairs. Given the limited number of adjectives
available in the heldout set, such swapping process can lead to certain heldout pairs dominating the
split. To address this, we identify these dominant pairs and try to avoid introducing them if there are
other candidates in the caption that can be swapped. Even with such measure, the frequencies of
