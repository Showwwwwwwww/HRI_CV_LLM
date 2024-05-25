from client2 import *
import argparse


if __name__ == '__main__':
      # Create the parser
    parser = argparse.ArgumentParser(description="Set CUDA device number")

    # Add arguments
    parser.add_argument('--device', type=int, required=True, help='CUDA device number')
     # Add arguments
    parser.add_argument('--scenario', type=str, required=True, help='scenario in the conversation')


    # Parse the arguments
    args = parser.parse_args()

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    print(f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

    C = Client(device = args.device,scenario=args.scenario)
    C.communicate_behavior2()
    C.shutdown()