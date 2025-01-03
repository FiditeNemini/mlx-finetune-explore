import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="path to the input data")
parser.add_argument("--output", "-o", required=True, help="path to the output file")

args = parser.parse_args()


def main():
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    
    content : str
    with open(args.input, "r") as in_file:
        content = in_file.read()
    
    with open(args.output, "w") as out_file:
        out_file.write(content)

if __name__ == "__main__":
    main()
