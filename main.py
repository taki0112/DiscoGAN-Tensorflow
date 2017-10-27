
from DiscoGAN import DiscoGAN
import argparse
from ops import *
from utils import *
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of DiscoGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning_rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight_decay')
    parser.add_argument('--dataset', type=str, default='cat2dog', help='dataset_name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = DiscoGAN(sess, epoch=args.epoch, dataset=args.dataset, batch_size=args.batch_size, learning_rate=args.lr,
                       beta1=args.beta1, beta2 =args.beta2, weight_decay=args.weight_decay,
                       checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir, sample_dir=args.sample_dir)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()