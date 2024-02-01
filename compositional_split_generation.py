"""
This code produces:
	>> split.pkl: contains lists of image ids (uids or uris) for train, test_seen, and test_unseen. It also contains information about the heldout pairs in <adj>_<nn> formatted string as a set
	>> data.pkl: contains original caption and swqapped captions retrievable by image ids
	
The formats of these outputs are compatible with the original CLIP-R Precision implementation comp-t2i-dataset found here: https://github.com/Seth-Park/comp-t2i-dataset/tree/main
"""

import os
import json
import spacy
import pickle
import random
import numpy as np

import argparse

def main():
	parser = argparse.ArgumentParser()
	
	# /home/ptclient/text_guided_3D_gen/TAPS3D/data/human_captions/chair/id_captions.json
	parser.add_argument('--caption_path', default='id_captions.json', help='path to caption')
	
	# ./clip_r_precision
	parser.add_argument('--save_path', default='./pickles', help='directory where the pickle files should be saved')
	
	# CONFIG HERE: remove the below arguments if it does not suit your dataset
	parser.add_argument('--dataset', default='shapenet', help='type of dataset the caption is labeling')
	parser.add_argument('--category', default='chair', help='which category the captions are describing')
	parser.add_argument('--caption_type', default='human_captions', help='choose from [human_captions, pseudo_captions, capdec_captions]')
	
	args = parser.parse_args()
	
	if args.dataset == 'shapenet':
	
		# configure this path so that it points to the correct json caption file
		caption_fp = args.caption_path
		# configure this path so that it points to the directory where the data and split pickles should be saved
		save_dir = args.save_path
		if not os.path.isdir(save_dir): os.mkdir(save_dir)
		
		# CONFIG HERE: determine how the pickles should be organized in the save_path directory
		caption_type_dir = os.path.join(save_dir, args.caption_type)
		if not os.path.isdir(caption_type_dir): os.mkdir(caption_type_dir)
		
		category_dir = os.path.join(caption_type_dir, args.category)
		if not os.path.isdir(category_dir): os.mkdir(category_dir)
		
		# load caption json as python dict
		with open(caption_fp, 'r') as f:
			caption_data = json.load(f)
			
		# set up Spacy
		# install in terminal with: 
		# 	python -m spacy download en_core_web_sm
		nlp = spacy.load('en_core_web_sm')
		
		# load color and shape adjectives from text files
		with open('./colors.txt', 'r') as f:
			color_adjs = f.read().split('\n')
		with open('./shapes.txt', 'r') as f:
			shape_adjs = f.read().split('\n')

		# set up adjective-noun frequencies tracker
		# first level key: 
		# 	<adjective>_<lemmatized noun>
		# first level value: 
		#	list of dict objects
		# second level keys: 
		# 	1. uid: 			model unique id in ShapeNet
		# 	2. caption id: 	0-based index converted to str
		# 	3. adj: 			original casing
		# 	4. nn: 			original casing, before lemmatization
		adj_nn_freq = {}

		# first level key: adjective, lowercase
		# value: frequencies
		adj_freq = {}

		# set up dict that will eventually be data.pkl
		data_pkl = {}
		
		# CONFIG HERE: change code here to loop through the captions in your dataset

		# loop through the captions
		uids = caption_data.keys()
		print(f'len(uids): {len(uids)}')
		for uid in uids:
		    
		    # CONFIG HERE: make sure variable caption is a string that contains a single caption
			model_data = caption_data[uid]
			caption = model_data[0]
			doc = nlp(caption)
			
			# store in data_pkl
			if uid not in data_pkl.keys():
				data_pkl[uid] = {}
			caption_id = len(data_pkl[uid])
			data_pkl[uid][caption_id] = {'text': caption}
			
			
			# find adj-noun pairs by frequencies
			for i, token in enumerate(doc):
				if i == len(doc)-1: continue 			# make sure token is adjective
				if token.pos_ != 'ADJ': continue		# make sure token has next
				next_token = doc[i+1]
				if next_token.pos_ != 'NOUN': continue	# make sure next token is noun
				
				# standardize adjective with lowercase convertion
				adj = token.text.lower()
				
				# track adjective frequencies
				if adj not in adj_freq.keys():
					adj_freq[adj] = 1
				else:
					adj_freq[adj] += 1
				
				# standardize noun by lemmatization and lowercase convertion
				lemma_noun = next_token.lemma_
				
				adj_nn_key = adj + '_' + lemma_noun
				if adj_nn_key not in adj_nn_freq.keys():
					adj_nn_freq[adj_nn_key] = [{'uid': uid, 
												'caption_id': caption_id, 
												'adj': token.text, 
												'nn': next_token.text}]
				else:
					adj_nn_freq[adj_nn_key].append({'uid': uid, 
													'caption_id': caption_id, 
													'adj': token.text, 
													'nn': next_token.text})

		# determine heldout pairs
		adj_nn_freq_list = sorted(adj_nn_freq, key=lambda k: len(adj_nn_freq[k]), reverse=True)
		percentile_25 = int((len(adj_nn_freq_list) / 100) * 25)
		percentile_75 = int((len(adj_nn_freq_list) / 100) * 75) + 1
		heldout_pairs = adj_nn_freq_list[percentile_25:percentile_75]
		heldout_pairs = random.sample(heldout_pairs, int(len(adj_nn_freq_list) / 10))
		print(adj_nn_freq_list)
		print(f'len(adj_nn_freq_list): {len(adj_nn_freq_list)}')
		print(f'len(heldout_pairs): {len(heldout_pairs)}')
		
		# create a list of test_unseen uids
		test_unseen = set()
		# add heldout_pairs key:value pair to data_pkl
		for pair in heldout_pairs:
			occurrences = adj_nn_freq[pair]
			for o in occurrences:
				uid = o['uid']
				caption_id = o['caption_id']
				
				test_unseen.add(uid)
				
				if 'heldout_pairs' not in data_pkl[uid][caption_id].keys():
					data_pkl[uid][caption_id]['heldout_pairs'] = [pair]
				else:
					data_pkl[uid][caption_id]['heldout_pairs'].append(pair)

		# create test_seen by first excluding test_unseen
		test_seen = set(caption_data.keys()).difference(test_unseen)
		# make sure test_seen and test_unseen are the same size
		test_seen = set(random.sample(list(test_seen), len(test_unseen)))

		# create training set by excluding both test_seen and test_unseen
		train = set(caption_data.keys()).difference(test_seen)
		train = train.difference(test_unseen)

		# configure split.pkl
		split_pkl = {
			'train': list(train), 
			'test_seen': list(test_seen), 
			'test_unseen': list(test_unseen), 
			'heldout_pairs': set(heldout_pairs)
		}

		with open(os.path.join(save_dir, f'{args.caption_type}/{args.category}/data.pkl'), 'wb') as f:
			pickle.dump(split_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

		# find the 60 most frequent adjectives
		top_60_adjs = sorted(adj_freq, key=lambda k: adj_freq[k], reverse=True)
		# filter out adjectives that aren't in color or shape adjectives list
		top_60_adjs = [adj for adj in top_60_adjs if adj in color_adjs or adj in shape_adjs]

		# find top 100 most frequent adjective-noun pairs based on the top 60 adjectives
		top_100_pairs = {}
		for i in range(100):
			top_100_pairs[f'<adj_nn_{i}>'] = 0
		
		for adj_nn_key in adj_nn_freq_list:
			if adj_nn_key.split('_')[0] not in top_60_adjs: continue
			
			# get frequency of this adj_nn pair
			this_adj_nn_freq = len(adj_nn_freq[adj_nn_key])
			
			# sort from high frequency to low frequency
			top_100_pairs_sorted_keys = sorted(top_100_pairs, key=top_100_pairs.get, reverse=True)
			
			# check if it is higher than the lowest frequency pair in top_100
			if top_100_pairs[top_100_pairs_sorted_keys[-1]] < this_adj_nn_freq:
				del top_100_pairs[top_100_pairs_sorted_keys[-1]]
				top_100_pairs[adj_nn_key] = this_adj_nn_freq

		# sort one more time from high frequency to low frequency
		top_100_pairs_sorted_keys = sorted(top_100_pairs, key=top_100_pairs.get, reverse=True)
		print(f'sorted top 100 adj-noun pairs: {top_100_pairs}')

		# we first identify the dorminant pairs in the top 100 adjective-noun pairs
		# by selecting the pairs in the 25th percentile of top_100_pairs
		top_100_25th_percentile_freq = np.percentile(list(top_100_pairs.values()), 25)
		top_100_above_25th = []
		for key in list(top_100_pairs.keys()):
			if top_100_pairs[key] > top_100_25th_percentile_freq:
				top_100_above_25th.append(key)

		# for swapping adjectives
		heldout_pairs_adj = [pair.split('_')[0] for pair in heldout_pairs]
		# we produce swapped captions from test_seen
		for uid in test_seen:
			caption_objs = data_pkl[uid]
			for caption_id in caption_objs.keys():
				ori_caption = caption_objs[caption_id]['text']
				
				# find adjective-noun pairs
				adj_nn_pairs = []
				adj_nn_pairs_freq = []
				ori_adjs = []
				doc = nlp(caption)
				for i, token in enumerate(doc):
					if i == len(doc)-1: continue 			# make sure token is adjective
					if token.pos_ != 'ADJ': continue		# make sure token has next
					next_token = doc[i+1]
					if next_token.pos_ != 'NOUN': continue	# make sure next token is noun
				
					# standardize adjective with lowercase convertion
					adj = token.text.lower()
					lemma_noun = next_token.lemma_
					
					ori_adjs.append(token.text)
					adj_nn_pairs.append(adj + '_' + lemma_noun)
					
				
				if len(adj_nn_pairs) == 0:
					data_pkl[uid][caption_id]['swapped_text'] = ori_caption
					data_pkl[uid][caption_id]['changes_made'] = {
						'noun': "none", 
						'original_adj': "none",
						'new_adj': "none"
					}
				elif len(adj_nn_pairs) == 1:
					ori_adj = ori_adjs[0].lower()
					new_adj = ori_adj
					while new_adj == ori_adj:
						new_adj = heldout_pairs_adj[random.randint(0, len(heldout_pairs_adj)-1)]
					data_pkl[uid][caption_id]['swapped_text'] = ori_caption.replace(ori_adjs[0], new_adj)
					data_pkl[uid][caption_id]['changes_made'] = {
						'noun': adj_nn_pairs[0].split('_')[-1], 
						'original_adj': ori_adj,
						'new_adj': new_adj
					}
				else: # more than one option for swapping
				
					# check if adj_nn_pairs in 25th percentile of top_100_pairs
					if set(adj_nn_pairs).issubset(top_100_above_25th):
						# adj_nn_pairs is a subset of top_100_above_25th
						# every option is in the 25th percentile
						# no adjustments can be made
						noun = adj_nn_pairs[0].split('_')[-1]
						ori_adj = ori_adjs[0].lower()
						new_adj = ori_adj
						while new_adj == ori_adj:
							new_adj = heldout_pairs_adj[random.randint(0, len(heldout_pairs_adj)-1)]
					else:
						# find the option that is not in 25th percentile of top_100_pairs
						ori_adj = ori_adjs[0].lower()
						new_adj = ori_adj
						
						adj_noun_candidates = []
						for pair in adj_nn_pairs:
							if pair not in top_100_above_25th:
								adj = pair.split('_')[0]
								if adj != ori_adj:
									adj_noun_candidates.append(adj + '_' + pair.split('_')[-1])
						adj_noun_candidate = adj_noun_candidates[random.randint(0, len(adj_noun_candidates)-1)]
						new_adj = adj_noun_candidate.split('_')[0]
						noun = adj_noun_candidate.split('_')[1]
					
					data_pkl[uid][caption_id]['swapped_text'] = ori_caption.replace(ori_adjs[0], new_adj)
					data_pkl[uid][caption_id]['changes_made'] = {
						'noun': noun, 
						'original_adj': ori_adj,
						'new_adj': new_adj
					}

		# configrue data.pkl
		with open(os.path.join(save_dir, f'{args.caption_type}/{args.category}/data.pkl'), 'wb') as f:
			pickle.dump(data_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)		


if __name__ == '__main__':
	main()






