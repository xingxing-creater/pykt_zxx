import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="lefokt_akt")
    # /qid_noforgetting/qid_ailibi/qid_log/qid_power/qid_fire/qid_t5/qid_sandwich
    parser.add_argument("--emb_type", type=str, default="qid") 
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--num_buckets", type=int, default=16)
    parser.add_argument("--max_distance", type=int, default=100)
    
    parser.add_argument("--init_c", type=float, default=0.01)
    parser.add_argument("--init_L", type=int, default=50)
    
    parser.add_argument("--bar_d", type=int, default=32)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    

    args = parser.parse_args()

    params = vars(args)
    main(params)
