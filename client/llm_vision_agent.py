from client2 import *
import argparse
import os

if __name__ == '__main__':
      # Create the parser
    # parser = argparse.ArgumentParser(description="Set CUDA device number")

    # # Add arguments
    # parser.add_argument('--device', type=int, default=1,required=True, help='CUDA device number')
    #  # Add arguments
    # parser.add_argument('--scenario', type=str, required=True, help='scenario in the conversation')


    # # Parse the arguments
    # args = parser.parse_args()

    # Set the CUDA_VISIBLE_DEVICES environment variable
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    #print(f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
    #print('device set to ',str(args.device))
    #print(f"scenario set to {args.scenario}")
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    C = Client(device = 1)
    C.llm_vision_agent(person = 'ShuoChen',scenario='Education')
    C.shutdown()