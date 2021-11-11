from datasets import load_dataset
import torch
import pickle
import os
import pdb
from transformers import AutoTokenizer

# here, squad means squad2

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_name, tokenizer,  type, logger):
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.logger = logger

        try:
            logger.info("Load processed data")
            with open(f'data/preprocessed_{type}_{data_name}.pickle', 'rb') as f:
                encodings = pickle.load(f)
        except:
            logger.info("preprocessing data...")
            raw_dataset = load_dataset(self.data_name)
            context, question, answer = self._preprocessing_dataset(raw_dataset[type])
            assert len(context) == len(question) == len(answer['answer_start']) == len(answer['answer_end'])

            logger.info("Encoding dataset (it will takes some time)")
            encodings = tokenizer(question, context, truncation='only_second', padding=True) # [CLS] question [SEP] context
            logger.info("add token position")
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
        self.logger.info(f"preprocessing {self.data_name} data")
        if self.data_name == "mrqa":
            for i, (c, q, ans) in enumerate(zip(dataset['context'], dataset['question'],dataset['answers'])):
                for a in ans:
                    context.append(c)
                    question.append(q)
                    answer['answer_start'].append(c.find(a) )
                    answer['answer_end'].append(c.find(a) + len(a) )

        elif self.data_name == 'coqa':
            # Can't use
            for i, (c,qs,ans) in enumerate(zip(dataset['story'],dataset['questions'],dataset['answers'])):
                for i in range(len(qs)):
                    context.append(c)
                    question.append(qs[i])
                    answer['answer_start'].append(ans['answer_start'][i]) # out of dated.  [CLS] + question + [SEP]
                    answer['answer_end'].append(ans['answer_end'][i])
        
        elif self.data_name == "squad":
            # Can't use
            for i, (c, q, ans) in enumerate(zip( dataset['context'], dataset['question'], dataset['answers'])):
                for s_idx, text in zip(ans['answer_start'],ans['text']):# out of dated.  [CLS] + question + [SEP]
                    context.append(c)
                    question.append(q)
                    answer['answer_start'].append(s_idx)
                    answer['answer_end'].append(s_idx + len(text))
        
        else :
            self.logger.info("wrong dataset name")
            exit()
        return context, question, answer

    def _char_to_token_with_possible(self, i, encodings, char_position, type):
        if type == 'start':
            possible_position = [0,-1,1]
        else:
            possible_position = [-1,-2,0]

        for pp in possible_position:
            position = encodings.char_to_token(i, char_position + pp, sequence_index = 1) 
            if position != None:
                position +=1
                break
        
        return position

        # self.tokenizer.convert_tokens_to_string(encodings[i].tokens)
    def _add_token_positions(self, encodings, answers):
        # convert_ids_to_tokens
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
                # start_positions[-1] = self.tokenizer.model_max_length
                start_positions[-1] = 0
            if end_positions[-1] is None:
                # end_positions[-1] = self.tokenizer.model_max_length
                end_positions[-1] = 0
                

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       



def main():
    print("Testing dataset.py")
    makedirs("./data"); makedirs("./logs"); makedirs("./model");
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True,return_offsets_mapping = True)
    val_dataset = Dataset('mrqa', tokenizer,  "validation") 
    
    
if __name__ == '__main__':
    main()