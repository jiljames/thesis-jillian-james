import random


PASSIVE_EATING_VERBS = ["eaten", "consumed", "devoured"]
ACTIVE_EATING_VERBS = ["eats", "consumes", "devours"]


def generate_random_passive_sent(food_vocab_size, eater_vocab_size):
    sent = "food" + str(random.randint(1, food_vocab_size))
    sent += " is " + random.choice(PASSIVE_EATING_VERBS)
    sent += " by eater" + str(random.randint(1, eater_vocab_size))
    return sent

def generate_random_active_sent(food_vocab_size, eater_vocab_size):
    sent = "eater" + str(random.randint(1, eater_vocab_size))
    sent += " " + random.choice(ACTIVE_EATING_VERBS)
    sent += " food" + str(random.randint(1, food_vocab_size))
    return sent

def generate_random_sent(food_vocab_size, eater_vocab_size):
    if random.random() < 0.5:
        return generate_random_active_sent(food_vocab_size, eater_vocab_size)
    else:
        return generate_random_passive_sent(food_vocab_size, eater_vocab_size)

def generate_random_sents(train_file, test_file, num_sents,
                          food_vocab_size, eater_vocab_size):
    sents = set()
    for i in range(num_sents):
        sents.add(generate_random_sent(food_vocab_size, eater_vocab_size))
    sents = list(sents)
    split_point = len(sents)//2 
    with open(train_file, 'w') as writer:
        for sent in sents[:split_point]:
            writer.write(sent)
            writer.write('\n')
    with open(test_file, 'w') as writer:
        for sent in sents[split_point:]:
            writer.write(sent)
            writer.write('\n')
    
