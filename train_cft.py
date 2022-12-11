from train_utils import get_data, train_model, save_checkpoint_callback
import argparse
import datetime 
import time
import sys
import os


# python -u train_cft.py /Users/abuj/Documents/GitHub/CFT/ day test1 --lw 7 --slw 21 --pct-train 0.85 --batch-size 64 --n-epochs 10 --repr-dims 320 

if __name__ == "__main__":

    sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', help='The base path to get the data')
    parser.add_argument('timeframe', help='The time frame to use for training day or hour')
    parser.add_argument('model_name', help='The model name')

    parser.add_argument('--lw', type=int,  default=7, help='The lookback window')
    parser.add_argument('--slw', type=int,  default=21, help='The standardize lookback window')
    parser.add_argument('--pct-train', type=float,  default=0.85, help='The standardize lookback window')
    parser.add_argument('--n-epochs', type=int, default=10, help='The number of epochs')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--batch-size', type=int, default=64, help='The batch size (defaults to 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')


    # parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    # parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    # parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    # parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    # parser.add_argument('--seed', type=int, default=None, help='The random seed')
    # parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    # parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()

    print("Arguments:", str(args))
    
    data_dict, train_data, exp_train_data, test_data, exp_test_data = get_data(args.base_path, args.timeframe, lookback_window=args.lw, standardize_lookback_window=args.slw, pct_train=args.pct_train)
  
    os.makedirs("saved_models/"+args.model_name, exist_ok=True)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        n_epochs=args.n_epochs, 
        device=0 #TODO: Update this
    )

    if args.save_every is not None:
        unit = 'epoch' if args.n_epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback("saved_models/"+args.model_name, args.save_every, unit)

    t = time.time()
    # train_model(train_data, exp_train_data, args.model_name, **config)
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")



