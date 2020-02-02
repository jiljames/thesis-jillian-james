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

def generate_random_sents(out_file, num_sents,
                          food_vocab_size, eater_vocab_size):
    sents = set()
    for i in range(num_sents):
        sents.add(generate_random_sent(food_vocab_size, eater_vocab_size))
    sents = list(sents)
    with open(out_file, 'w') as writer:
        for sent in sents:
            writer.write(sent)
            writer.write('\n')
    

def is_valid_passive(phrase):
    if len(phrase) != 5: return False
    if "food" not in phrase[0]: return False
    if phrase[1] != "is": return False
    if phrase[2] not in PASSIVE_EATING_VERBS: return False
    if phrase[3] != "by": return False
    if "eater" not in phrase[4]: return False
    return True


def is_valid_active(phrase):
    if len(phrase) != 3: return False
    if "eater" not in phrase[0]: return False
    if phrase[1] not in ACTIVE_EATING_VERBS: return False
    if "food" not in phrase[2]: return False
    return True

def is_valid_phrase(phrase):
    return is_valid_active(phrase) or is_valid_passive(phrase)