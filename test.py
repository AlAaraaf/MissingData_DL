import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type = int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.id)