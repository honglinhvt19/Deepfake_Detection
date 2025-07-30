import argparse
from training.train import train
from training.evaluate import evaluate
from training.inference import inference

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], required=True,
                        help="Mode to run: train, evaluate, or inference")
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help="Path to config file")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.h5',
                        help="Path to model checkpoint (for evaluate/inference)")
    parser.add_argument('--video', type=str, help="Path to video file (for inference)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'evaluate':
        evaluate(args.config, args.checkpoint)
    elif args.mode == 'inference':
        if not args.video:
            raise ValueError("Video path is required for inference mode")
        inference(args.video, args.config, args.checkpoint)

if __name__ == "__main__":
    main()