import argparse
from wandb_train4promptkt import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pretrain")
    parser.add_argument("--not_select_dataset", type=str, default="all")
    parser.add_argument("--re_mapping", type=str, default="0")
    parser.add_argument("--model_name", type=str, default="promptkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--final_fc_dim", type=int, default=64)
    parser.add_argument("--final_fc_dim2", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--loss1", type=float, default=0.5)
    parser.add_argument("--loss2", type=float, default=0.5)
    parser.add_argument("--loss3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # multi-task
    parser.add_argument("--cf_weight", type=float, default=0.1)
    parser.add_argument("--t_weight", type=float, default=0.1)

    parser.add_argument("--seq_len", type=int, default=200)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--global_bs", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=1.0)

    parser.add_argument("--pretrain_path", type=str, default="")
    parser.add_argument("--pretrain_epoch", type=int, default=0)
    parser.add_argument("--train_mode", type=str, default="pretrain")
    parser.add_argument("--project_name", type=str, default="promptkt")

    args = parser.parse_args()

    params = vars(args)
    main(params, args)
