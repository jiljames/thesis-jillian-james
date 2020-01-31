from sklearn.model_selection import train_test_split
import sys
import re
from os.path import join


class HaikuTokenizer:
    
    def __init__(self, desired_seg_len):
        self.desired_seg_len = desired_seg_len
    
    def __call__(self, lines):
        result = []
        segment = []
        for line in lines:
            if line == "\n": 
                result.append(self.finalize_segment(segment))
                segment = []
            else:
                segment += self.tokenize_line(line)
        return result
    
    def finalize_segment(self, segment):
        num_stopwords = self.desired_seg_len - len(segment)
        if num_stopwords > 0:
            segment += [" _FILL_ "] * num_stopwords
        else:
            segment = segment[:self.desired_seg_len]
        return segment
    
    def tokenize_line(self, line):
        line = line.strip()
        line = list(line.lower())
        line.append(" _BREAK_ ")
        return line

class ObamaTokenizer:

    def __init__(self, desired_seg_len):
        self.desired_seg_len = desired_seg_len
    
    def __call__(self, lines):
        full = []
        paragraph = []
        n = 0
        for line in lines:
            if line != "\n":
                line = line.strip()
                line = re.findall(r"[\w']+|[.,!?();-]", line.lower())
                num_stopwords = self.desired_seg_len - len(line)
                if line != []:
                    if len(line) < self.desired_seg_len:
                        paragraph = line + [" _FILL_ "]*num_stopwords
                        full.append(paragraph)
                    else:
                        paragraph = line[:self.desired_seg_len]
                        full.append(paragraph)
                    n+= 1
        return full


class Vocab:
    
    def __init__(self, all_tokens):
        self.all_tokens = all_tokens
        self.int_to_word, self.word_to_int = {}, {}
    
        for i in range(len(all_tokens)):
            self.word_to_int[all_tokens[i]] = i
            self.int_to_word[i] = all_tokens[i]
    
    def __len__(self):
        return len(self.all_tokens)

    def encode(self, text_ls):
        new_full = []
        for text in text_ls:
            new_text = []
            for word in text:
                new_text.append(self.word_to_int[word])
            new_full.append(new_text)
        return new_full
    
    def decode(self, lines):
        text_ls = []    
        for line in lines:
            parse_line = [self.int_to_word[x] for x in line 
                          if self.int_to_word[x] != " _FILL_ "]
            parse_line = " ".join(parse_line)
            text_ls.append(parse_line)
        return text_ls

    @staticmethod
    def construct(lines_ls):        
        all_tokens= []
        for line in lines_ls:
            for token in line:
                all_tokens.append(token)
        all_tokens = list(set(all_tokens))
        all_tokens = sorted(all_tokens)
        return Vocab(all_tokens)



class Task:
    def __init__(self, vocab, path, max_seq_length, num_train, 
                 num_valid, num_test):
        self.name = ""
        self.vocab = vocab
        self.path = path
        self.train_file = join(path, "encoded.train.txt")
        self.valid_file = join(path, "encoded.valid.txt")
        self.test_file = join(path, "encoded.test.txt")
        self.log_file = join(path, "log.txt")
        self.negative_file = join(path, "negative.txt")
        self.eval_file = join(path, "eval_file.txt")
        self.max_seq_length = max_seq_length
        self.vocab_size = len(vocab)
        self.generated_num = num_train
        self.num_valid = num_valid
        self.num_test = num_test


def load_task(taskname):
    def write_lists_to_file(filename, full_lists):
        with open(filename, "w") as f:
            for ls in full_lists:
                line = " ".join([str(x) for x in ls])+ "\n"
                f.write(line)
    self.name = taskname
    print("Loading task: " + str(taskname))
    if taskname == 'obama':
        max_seq_length = 40
        tokenizer = ObamaTokenizer(max_seq_length)
    elif taskname == 'synth':
        max_seq_length = 20
        tokenizer = ObamaTokenizer(max_seq_length)        
    elif taskname == 'haiku': 
        max_seq_length = 70
        tokenizer = HaikuTokenizer(max_seq_length)
    else:
        print("Application must be haiku or obama or synth")
        sys.exit(0)
     
    path = join("../data", taskname)
    input_file = join(path, "input.txt")         
    whole = tokenizer(open(input_file, 'r'))
    train_ls, remainder = train_test_split(whole, test_size = 0.4, 
                                           shuffle = False)
    valid_ls, test_ls = train_test_split(remainder, test_size = 0.5, 
                                         shuffle = False)

    # Create dictionaries to map integers to tokens and vice versa
    vocab = Vocab.construct(train_ls + valid_ls + test_ls)
    train_as_int_ls = vocab.encode(train_ls)
    valid_as_int_ls = vocab.encode(valid_ls)
    test_as_int_ls = vocab.encode(test_ls)

    task = Task(vocab, path, max_seq_length, len(train_ls), 
                len(valid_ls), len(test_ls))

    # Write to correct application training and validation files
    # Write conversion dictionaries

    write_lists_to_file(task.train_file, train_as_int_ls)
    write_lists_to_file(task.valid_file, valid_as_int_ls)
    write_lists_to_file(task.test_file, test_as_int_ls)
    
    print("Task successfully loaded.")

    return task

