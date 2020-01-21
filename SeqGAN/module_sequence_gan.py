import numpy as np
import tensorflow as tf
import random
import time
import os
import datautil
import nltk
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator, inspect_samples
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle
import argparse
import json

# For me for running on Mark's machine
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" 

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
generated_num = 10000 

#########################################################################################
#  Grouping Files for Parser
#########################################################################################
synth_files = {}
synth_files["log_file"] =  "synth/text_log.txt"
synth_files["positive_file"] =  'synth/text_to_int.train.txt'
synth_files["negative_file"] = 'synth/generator_sample.txt'
synth_files["test_file"] = "synth/text_to_int.test.txt"
synth_files["valid_file"] = "synth/text_to_int.valid.txt"
synth_files["eval_file"] =  'synth/eval_file.txt'
synth_files["int2word"] =  "synth/int_to_word.json"

obama_files = {}
obama_files["log_file"] =  "obama/obama_log.txt"
obama_files["positive_file"] =  'obama/obama_to_int.train.txt'
obama_files["negative_file"] = 'obama/generator_sample.txt'
obama_files["valid_file"] = "obama/obama_to_int.valid.txt"
obama_files["test_file"] = "obama/obama_to_int.test.txt"
obama_files["eval_file"] =  'obama/eval_file.txt'
obama_files["int2word"] =  "obama/int_to_word.json"

haiku_files = {}
haiku_files["log_file"] =  "haiku/haiku_log.txt"
haiku_files["positive_file"] = 'haiku/haiku_to_int.train.txt'
haiku_files["negative_file"] = 'haiku/generator_sample.txt'
haiku_files["valid_file"] = "haiku/haiku_to_int.valid.txt"
obama_files["test_file"] = "haiku/haiku_to_int.test.txt"
haiku_files["eval_file"] =  'haiku/eval_file.txt'
haiku_files["int2word"] =  "haiku/int_to_word.json"

#  Create a parser to parse user input
def create_parser():
    parser = argparse.ArgumentParser(description='Program for running several SeqGan applications.')
    parser.add_argument('app', metavar='application', type=str, choices=['obama', 'haiku', 'synth'],
                    help='Enter either \'obama\' or \'haiku\' or \'synth\'')
    parser.add_argument('gen_n', type = int,
                    help='Number of generator pre-training steps')
    parser.add_argument('disc_n', type = int,
                    help='Number of discriminator pre-training steps')
    parser.add_argument('adv_n', type = int,
                    help='Number of adversarial pre-training steps')
    parser.add_argument('-l', metavar="seq_len", type = int, default = -1,
                    help = 'Length of the token sequences used for training.')
    parser.add_argument('-v', metavar="vocab_size", type = int, default = -1,
                    help = "The size of the vocab from the input files (outout by datautil.py)")
    parser.add_argument('-mn', metavar="model_name", type = str, default = "",
                    help = "Name for the checkpoint files. Will be stored at ./<app>/models/<model_name>")

    return parser

def assign_parser_args(args):
    # Need to add functionality to allow user-specified N to be used in training
    if args.app == 'haiku':
        if args.l == -1:
            args.l = 70
        if args.vocab_size == -1:
            args.v = 60
        files = haiku_files
    elif args.app == 'obama':
        if args.l == -1:
            args.l = 40
        if args.v == -1:
            args.v = 13439
        files = obama_files
    else:
        if args.l == -1:
            args.l = 40
        if args.v == -1:
            args.v = 209
        files = synth_files
        

    #Make the /models directory if its not there.
    model_string = args.app +"/models/"
    if not os.path.exists("./"+model_string):
        os.mkdir("./"+model_string)

    #make the checkpoint directory if its not there.
    if args.mn == "":
        model_string += str(args.gen_n)+ "_" + str(args.disc_n) + "_" + str(args.adv_n)
        model_string += time.strftime("_on_%m_%d_%y", time.gmtime())
    else:
        model_string += args.mn
    if not os.path.exists("./"+model_string):
        os.mkdir("./"+model_string)

    return files, args.v, args.l, args.gen_n, args.disc_n, args.adv_n, model_string

#   Modularized Training

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_generator(sess, saver, MODEL_STRING, generator, gen_data_loader, likelihood_data_loader, files, log, num_epochs):
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(num_epochs):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, files["eval_file"])
            likelihood_data_loader.create_batches(files["valid_file"])
            small_loss = target_loss(sess, generator, likelihood_data_loader)
            saver.save(sess, MODEL_STRING+"/model")
        if epoch % 1 == 0:
            with open(files['int2word']) as json_file:
                int_to_word = json.load(json_file)
                int_to_word = {int(k): int_to_word[k] for k in int_to_word}
            inspect_samples(sess, generator, BATCH_SIZE, 3, int_to_word)
            generate_samples(sess, generator, BATCH_SIZE, generated_num, files["eval_file"])
            likelihood_data_loader.create_batches(files["valid_file"])
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            if test_loss < small_loss:
                small_loss = test_loss
                saver.save(sess, MODEL_STRING+"/model")
                print("Saving checkpoint ...")
            #else:
            #    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
            print('pre-train epoch {}: train_loss = {}; test_loss = {}'.format(epoch, loss, test_loss))
            buffer = 'epoch:\t'+ str(epoch) + '\tloss:\t' + str(loss) + '\n'
            log.write(buffer)
    return small_loss

def train_discriminator(sess, generator, discriminator, dis_data_loader, dis_test_data_loader, files, log, n):
    for i in range(n):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, files["negative_file"])
        dis_data_loader.load_train_data(files["positive_file"], files["negative_file"])
        dis_test_data_loader.load_train_data(files["valid_file"], files["negative_file"])
        losses = []
        test_losses = []
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                loss = sess.run(discriminator.train_op, feed)
                losses.append(loss)
            # for it in range(dis_test_data_loader.num_batch):
            #     x_batch, y_batch = dis_test_data_loader.next_batch()
            #     predicts_score= discriminator.score_predicts(sess, x_batch, y_batch)
            #     test_losses.append(predicts_score)

        print('train discriminator epoch {}: train_loss = {}, test_loss = {}'.format(i, np.mean(losses), np.mean(test_losses)))

def train_adversarial(sess, saver, MODEL_STRING, generator, discriminator, rollout, dis_data_loader, dis_test_data_loader, likelihood_data_loader, files, log, n):
    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    small_loss = float('inf')
    for total_batch in range(n):
        # Train the generator for one step
        samples = generator.generate(sess)
        rewards = rollout.get_reward(sess, samples, 16, discriminator) #I might actually need to change the value 16 here.
        feed = {generator.x: samples, generator.rewards: rewards}
        _ = sess.run(generator.g_updates, feed_dict=feed)
        
        # Test
        if total_batch % 1 == 0 or total_batch == n - 1:
            with open(files['int2word']) as json_file:
                int_to_word = json.load(json_file)
                int_to_word = {int(k): int_to_word[k] for k in int_to_word}
            inspect_samples(sess, generator, BATCH_SIZE, 3, int_to_word)
            generate_samples(sess, generator, BATCH_SIZE, generated_num, files["eval_file"])
            likelihood_data_loader.create_batches(files["positive_file"])
            loss = target_loss(sess, generator, likelihood_data_loader)
            likelihood_data_loader.create_batches(files["valid_file"])
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            if test_loss < small_loss:
                small_loss = test_loss
                saver.save(sess, MODEL_STRING +"/model")
                print("Saving checkpoint ...")
            # else:
            #     saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
            print("Adversarial Training total_batch: {}, train_loss: {}, test_loss: {}".format(total_batch, loss, test_loss))
            buffer = "total_batch: " + str(total_batch) + "test_loss: " + str(test_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator for 1 steps
        train_discriminator(sess, generator, discriminator, dis_data_loader, dis_test_data_loader, files, log, 1)


def main():
    #Get user input
    parser = create_parser()
    files, vocab_size, seq_length, gen_n, disc_n, adv_n, MODEL_STRING = assign_parser_args(parser.parse_args())


    # Initialize the random seed
    #random.seed(SEED)
    #np.random.seed(SEED)
    #tf.set_random_seed(SEED)
    assert START_TOKEN == 0

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Initialize the data loaders
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, seq_length)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, seq_length) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE, seq_length)
    dis_test_data_loader = Dis_dataloader(BATCH_SIZE, seq_length) # For testing

    # Initialize the Generator
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, seq_length, START_TOKEN)

    # Initialize the Discriminator
    discriminator = Discriminator(sequence_length=seq_length, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # Set session configurations. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # If restoring from a previous run ....
    if len(os.listdir("./"+MODEL_STRING)) > 0:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))


    # Create batches from the positive file.
    gen_data_loader.create_batches(files["positive_file"])

    # Open log file for writing
    log = open(files['log_file'], 'w')



    # Pre_train the generator with MLE. 
    pre_train_generator(sess, saver, MODEL_STRING, generator, gen_data_loader, likelihood_data_loader, files, log, gen_n)
    print('Start pre-training discriminator...')

    # Do the discriminator pre-training steps
    train_discriminator(sess, generator, discriminator, dis_data_loader, dis_test_data_loader, files, log, disc_n)
    
    # Do the adversarial training steps
    rollout = ROLLOUT(generator, 0.8)

    train_adversarial(sess, saver, MODEL_STRING, generator, discriminator, 
                    rollout, dis_data_loader, dis_test_data_loader, likelihood_data_loader, 
                    files, log, adv_n)

    #Use the best model to generate final sample
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    generate_samples(sess, generator, BATCH_SIZE, generated_num, files["eval_file"])

    # Calculate the BLEUscore
    int_to_word = json.load(open(files["int2word"], 'r'))
    generated = datautil.int_file_to_text_ls(open(files["eval_file"], 'r'), int_to_word)
    references = datautil.int_file_to_text_ls(open(files["test_file"], 'r'), int_to_word)

    generated = datautil.remove_filler(generated)
    references = datautil.remove_filler(references)

    blue = nltk.translate.bleu_score.corpus_bleu([references]*len(generated), generated)
    print("Run with args {} {} {}: BLEUscore = {}\n".format(gen_n, disc_n, adv_n, blue))
    

    if files == synth_files:
        
        total_correct = 0
        for sentence in generated:
            if datautil.is_phrase_valid_passive(sentence) or datautil.is_phrase_valid_active(sentence):
                total_correct +=1
        prop = total_correct/len(generated)
        
        if not os.path.exists("./synth/results.txt"):
            os.mknod("./synth/results.txt")

        with open("./synth/results.txt", 'a') as f:
            outblue = "synth run {} {} {}: BLEUscore = {}\n".format(gen_n, disc_n, adv_n, blue)
            f.write(outblue)
            out = "synth run {} {} {}: Proportion Valid = {}\n".format(gen_n, disc_n, adv_n, prop)
            f.write(out)
            f.close()

    log.close()


if __name__ == '__main__':
    main()
