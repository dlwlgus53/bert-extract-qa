from datasets import load_dataset
import torch
import pickle
from tqdm import tqdm
import pdb
# here, squad means squad2

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_name, tokenizer,  type):
        self.tokenizer = tokenizer
        self.data_name = data_name
        

        try:
            print("Load processed data")
            with open(f'data/preprocessed_{type}_{data_name}.pickle', 'rb') as f:
                encodings = pickle.load(f)
        except:
            print("preprocessing data...")
            raw_dataset = load_dataset(self.data_name)
            context, question, answer = self._preprocessing_dataset(raw_dataset[type])
            assert len(context) == len(question) == len(answer['answer_start']) == len(answer['answer_end'])

            print("Encoding dataset (it will takes some time)")
            encodings = tokenizer(context, question, truncation='only_first', padding=True) # [CLS] context [SEP] question
            print("add token position")
            self._add_token_positions(encodings, answer)

            with open(f'data/preprocessed_{type}_{data_name}.pickle', 'wb') as f:
                pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)

        self.encodings = encodings



    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    def _preprocessing_dataset(self, dataset):
        context = []
        question = []
        answer = {'answer_start' : [], 'answer_end' : []}
        print(f"preprocessing {self.data_name} data")
        if self.data_name == "mrqa":
            for i, (c, q, ans) in tqdm(enumerate(zip(dataset['context'], dataset['question'],dataset['answers'])), total= len(dataset['context'])):
                for a in ans:
                    context.append(c)
                    question.append(q)
                    answer['answer_start'].append(c.find(a))
                    answer['answer_end'].append(c.find(a) + len(a))

        elif self.data_name == 'coqa':
            for i, (c,qs,ans) in tqdm(enumerate(zip(dataset['story'],dataset['questions'],dataset['answers'])), total=len(dataset['story'])):
                for i in range(len(qs)):
                    context.append(c)
                    question.append(qs[i])
                    answer['answer_start'].append(ans['answer_start'][i])
                    answer['answer_end'].append(ans['answer_end'][i])
        
        elif self.data_name == "squad":
            for i, (c, q, ans) in tqdm(enumerate(zip( dataset['context'], dataset['question'], dataset['answers'])), total=len(dataset['context'])):
                for s_idx, text in zip(ans['answer_start'],ans['text']):
                    context.append(c)
                    question.append(q)
                    answer['answer_start'].append(s_idx)
                    answer['answer_end'].append(s_idx + len(text))
        
        else :
            print("wrong dataset name")
            exit()
        return context, question, answer

    def _char_to_token_with_possible(self, i, encodings, char_position, type):
        if type == 'start':
            possible_position = [0,-1,1]
        else:
            possible_position = [-1,-2,0]

        for pp in possible_position:
            position = encodings.char_to_token(i, char_position + pp)
            if position != None:
                break
        
        return position

        
    def _add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers['answer_start'])):
            # char의 index로 되어있던것을 token의 index로 찾아준다.
            if  answers['answer_start'][i] != -1: # for case of mrq
                start_char = answers['answer_start'][i] 
                end_char = answers['answer_end'][i]
                start_position = self._char_to_token_with_possible(i, encodings, start_char,'start')
                end_position = self._char_to_token_with_possible(i, encodings, end_char,'end')
                start_positions.append(start_position)
                end_positions.append(end_position)
            else:
                start_positions.append(None)
                end_positions.append(None)
            
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


