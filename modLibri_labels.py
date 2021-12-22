"""
Name: Riah Coulter
Date: 6-22-21
Purpose:
  Modify fairseq's original libri_labels.py which identifies letter and word
  labels for Wav2Vec2 experiments and stores that information in .ltr and .wrd
  files.

  Original problem: File conventions specifically for Librispeech formatting and
  not your own data. This program should work with the current wav2vec2_manifest.py
  script (that creates training/validation tsvs) and personal data instead of
  the Librispeech formatting. The key is to give wav2vec2_manifest.py one directory
  *above* the audio files directory rather than the specific directory as that
  sets up the tsv right so that this script can read it. I might simplify this in
  the future too.

  Parameters:
  -> directory to training or validationt tsv
  -> and output directory to save the files to
  -> an output name (such as 'train') for the files

  Update (7-1-21) added encoding=utf-8 to open files to prevent wrong encoding.
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w", encoding="utf-8"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w", encoding="utf-8"
    ) as wrd_out:
        root = next(tsv).strip()
        
        for line in tsv:
            line = line.strip()
            dir = os.path.dirname(line)

            if dir not in transcriptions:
                parts = dir.split(os.path.sep)

                trans_path = f"{parts[0]}.trans.txt"
                print(f"trans_path: {trans_path}")
                path = os.path.join(root, dir, trans_path)
                print(f"path: {path}")
                assert os.path.exists(path)
                texts = {}
                with open(path, "r") as trans_f:
                    for tline in trans_f:
                        items = tline.strip().split()
                        print(f"items: {items}")
                        texts[items[0]] = " ".join(items[1:])
                print(f"dir: {dir}")
                transcriptions[dir] = texts
            print(f"line: {line}")
            print(f"os.path.basename(line): {os.path.basename(line).split()}")
            part = os.path.basename(line).split()[0]
            print(f"part: {part}")
            print(f"transcriptions: {transcriptions}")
            assert part in transcriptions[dir]
            print(transcriptions[dir][part], file=wrd_out)
            print(
                " ".join(list(transcriptions[dir][part].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
