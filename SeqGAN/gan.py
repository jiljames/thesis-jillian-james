import tensorflow as tf
import time
import os
import csv
import argparse
from nltk.translate.bleu_score import corpus_bleu
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator, pre_train_generator
from discriminator import Discriminator, train_discriminator
from rollout import ROLLOUT
from trainutil import generate_samples, target_loss
from datautil import load_task
from synthetic import generate_random_sents, is_valid_phrase


###############################################################################
#  Generator  Hyper-parameters
###############################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 64

###############################################################################
#  Discriminator  Hyper-parameters
###############################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64


# For me for running on Mark's machine
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" 


def train_adversarial(sess, saver, MODEL_STRING, generator, discriminator, 
                      rollout, dis_data_loader, likelihood_data_loader, 
                      task, log, n):
    print('#################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    small_loss = float('inf')
    for total_batch in range(n):
        # Train the generator for one step
        samples = generator.generate(sess)
        rewards = rollout.get_reward(sess, samples, 16, discriminator) #I might actually need to change the value 16 here.
        feed = {generator.x: samples, generator.rewards: rewards}
        _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        sample = generate_samples(sess, generator, BATCH_SIZE, 
                                  task.generated_num, task.eval_file)
        print("Examples from generator:")
        for sample in task.vocab.decode(samples)[:5]:
            print(sample)

        likelihood_data_loader.create_batches(task.valid_file)
        test_loss = target_loss(sess, generator, likelihood_data_loader)
        if test_loss < small_loss:
            small_loss = test_loss
            saver.save(sess, MODEL_STRING +"/model")
            print("Saving checkpoint ...")
        print("total_batch: ", total_batch, "test_loss: ", test_loss)
        buffer = "total_batch: " + str(total_batch) + "test_loss: " + str(test_loss)
        log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator for 5 steps
        train_discriminator(sess, generator, discriminator, dis_data_loader, 
                            task, log, 5, BATCH_SIZE, task.generated_num,
                            dis_dropout_keep_prob)


def main():
    
    #  Create a parser to parse user input
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Program for running several SeqGan applications.')
        parser.add_argument('app', metavar='application', type=str, choices=['obama', 'haiku', 'synth'],
                        help='Enter either \'obama\' or \'haiku\'')
        parser.add_argument('gen_n', type = int,
                        help='Number of generator pre-training steps')
        parser.add_argument('disc_n', type = int,
                        help='Number of discriminator pre-training steps')
        parser.add_argument('adv_n', type = int,
                        help='Number of adversarial pre-training steps')
        parser.add_argument('-mn', metavar="model_name", type = str, default = "",
                        help = "Name for the checkpoint files. Will be stored at ./<app>/models/<model_name>")
        parser.add_argument('-numeat', metavar="num_eat", type = int, default = 500,
                        help = "For synthetic data generation. Determines number of eaters in vocab.")
        parser.add_argument('-numfeed', metavar="num_feed", type = int, default = 500,
                        help = "For synthetic data generation. Determines number of feeders in vocab.")
        parser.add_argument('-numsent', metavar="num_sent", type = int, default = 10000,
                        help = "For synthetic data generation. Determines number of sentences generated.")
        args = parser.parse_args()

        synth_gen_params = ("NA", "NA", "NA")
        if args.app == "synth":
            synth_gen_params = (args.numsent, args.numfeed, args.numeat)
            generate_random_sents("../data/synth/input.txt", args.numsent, args.numfeed, args.numeat)

        task = load_task(args.app)

        #Make the /models directory if its not there.
        model_string = task.path +"/models/"
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
    
        return args.gen_n, args.disc_n, args.adv_n, model_string, task, synth_gen_params
    
    gen_n, disc_n, adv_n, MODEL_STRING, task, SYNTH_GEN_PARAMS = parse_arguments()


    assert START_TOKEN == 0

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Initialize the data loaders
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, task.max_seq_length)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, task.max_seq_length) # For validation
    dis_data_loader = Dis_dataloader(BATCH_SIZE, task.max_seq_length)

    # Initialize the Generator
    generator = Generator(len(task.vocab), BATCH_SIZE, EMB_DIM, HIDDEN_DIM, 
                          task.max_seq_length, START_TOKEN)

    # Initialize the Discriminator
    discriminator = Discriminator(sequence_length=task.max_seq_length, 
                                  num_classes=2, 
                                  vocab_size=len(task.vocab), 
                                  embedding_size=dis_embedding_dim, 
                                  filter_sizes=dis_filter_sizes, 
                                  num_filters=dis_num_filters, 
                                  l2_reg_lambda=dis_l2_reg_lambda)

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
    gen_data_loader.create_batches(task.train_file)

    # Open log file for writing
    log = open(task.log_file, 'w')

    # Pre_train the generator with MLE. 
    pre_train_generator(sess, saver, MODEL_STRING, generator, gen_data_loader, 
                        likelihood_data_loader, task, log, gen_n, BATCH_SIZE,
                        task.generated_num)
    print('Start pre-training discriminator...')

    # Do the discriminator pre-training steps
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    train_discriminator(sess, generator, discriminator, dis_data_loader, 
                        task, log, disc_n, BATCH_SIZE, task.generated_num,
                        dis_dropout_keep_prob)
    print("Saving checkpoint ...")
    saver.save(sess, MODEL_STRING+ "/model")
    
    # Do the adversarial training steps
    rollout = ROLLOUT(generator, 0.8)
    train_adversarial(sess, saver, MODEL_STRING, generator, discriminator, 
                      rollout, dis_data_loader, likelihood_data_loader, 
                      task, log, adv_n)

    #Use the best model to generate final sample
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    generate_samples(sess, generator, BATCH_SIZE, task.generated_num, 
                     task.eval_file)


    #Writing results to CSV
    with open(task.eval_file) as f:
        generated = []
        for line in f:
            line = line.strip().split()
            generated.append(line)
        generated = task.vocab.decode(generated)
        f.close()

    with open(task.test_file) as f:
        references = []
        for line in f:
            line = line.strip().split()
            references.append(line)
        references = task.vocab.decode(references)  
        f.close()      

        
    if not os.path.exists("./results.csv"):
        os.mknod("./results.csv")

    with open("./results.csv", 'a') as csvfile:
        fieldnames = ["name", "task_name", "num_gen", "num_disc", "num_adv",
                    "num_sents", "num_feeders", "num_eaters", "BLEU", "prop_valid"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        csvfile.seek(0, os.SEEK_END) # go to end of file
        if not csvfile.tell(): # if current position is != 0)
            writer.writeheader()

        blue = corpus_bleu([references]*len(generated), generated)
        print("Run with args {} {} {}: BLEUscore = {}\n".format(gen_n, disc_n, adv_n, blue))
        
        prop = "NA"

        if task.name == "synth":
            total_correct = 0
            for sentence in generated:
                if is_valid_phrase(sentence):
                    total_correct +=1
            prop = total_correct/len(generated)

        writer.writerow({"name": MODEL_STRING, "task_name": task.name,  "num_gen": gen_n, 
                        "num_disc":disc_n, "num_adv": adv_n, "num_sents":SYNTH_GEN_PARAMS[0],
                        "num_feeders":SYNTH_GEN_PARAMS[1], "num_eaters":SYNTH_GEN_PARAMS[2],
                        "BLEU": blue, "prop_valid": prop})
        f.close()


    log.close()


if __name__ == '__main__':
    main()
